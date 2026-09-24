from pathlib import Path
import json, warnings, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data_frozen_v1/synthetic_learner_state_dataset_v1.0.csv'
SPLIT=ROOT/'results/data/train_test_split_ids.csv'
OUT=ROOT/'results/closed_loop'
OUT.mkdir(exist_ok=True)
SEED=42
rng=np.random.default_rng(SEED)

ACADEMIC=['age','gender','class_year','previous_gpa','prior_course_grade','attendance_rate','absence_hours','weekly_questions','quiz_average','assignment_average','assignment_on_time','midterm','practice_test_score']
BEHAVIORAL=['study_hours_week','lms_login_count','video_watch_minutes','time_management','self_regulation','recommendation_completion_rate']
PSYCHOLOGICAL=['motivation','self_efficacy','academic_anxiety','preferred_learning_type']
TOPIC=['conceptual_understanding','problem_solving','application_skills','critical_thinking','advanced_topics']
FULL=ACADEMIC+BEHAVIORAL+PSYCHOLOGICAL+TOPIC

TOPIC_TO_ACTIVITY={
 'Conceptual Understanding':'Concept Review',
 'Problem Solving':'Practice Questions',
 'Application Skills':'Application Exercise',
 'Critical Thinking':'Critical Thinking Activity',
 'Advanced Topics':'Advanced Practice'
}
ACTIVITY_TO_TOPIC={v:k for k,v in TOPIC_TO_ACTIVITY.items()}
ACTIVITY_TO_TOPIC['Guided Review']=None

TOPIC_COL={
 'Conceptual Understanding':'conceptual_understanding',
 'Problem Solving':'problem_solving',
 'Application Skills':'application_skills',
 'Critical Thinking':'critical_thinking',
 'Advanced Topics':'advanced_topics'
}

# Reproducible preprocessing: fit only on the 800-student training partition.
def prep(X):
    cat=X.select_dtypes(include=['object','category']).columns.tolist()
    num=[c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ])

df=pd.read_csv(DATA)
split=pd.read_csv(SPLIT)
ids=split.loc[split.partition=='test','student_id'].astype(int).tolist()
te=df[df.student_id.isin(ids)].copy().reset_index(drop=True)
tr=df[df.student_id.isin(split.loc[split.partition=='train','student_id'])].copy().reset_index(drop=True)
assert len(tr)==800 and len(te)==200

# Best models selected in Stage 3 from the frozen split.
models={
 'performance':Pipeline([('prep',prep(tr[FULL])),('model',Lasso(alpha=0.01,max_iter=20000))]),
 'risk':Pipeline([('prep',prep(tr[FULL])),('model',SVC(C=2.0,class_weight='balanced',probability=True,random_state=SEED))]),
 'topic':Pipeline([('prep',prep(tr[FULL])),('model',LogisticRegression(max_iter=5000))]),
 'recommendation':Pipeline([('prep',prep(tr[FULL])),('model',LogisticRegression(max_iter=5000))]),
 'gain':Pipeline([('prep',prep(tr[FULL])),('model',Lasso(alpha=0.01,max_iter=20000))]),
}
targets={'performance':'final_grade','risk':'learning_risk_level','topic':'weakest_topic','recommendation':'recommended_activity','gain':'expected_learning_gain'}
for k,p in models.items():
    p.fit(tr[FULL],tr[targets[k]])

# Model-based initial diagnostics on the held-out test set.
initial={
 'pred_final_grade':models['performance'].predict(te[FULL]),
 'pred_risk':models['risk'].predict(te[FULL]),
 'pred_topic':models['topic'].predict(te[FULL]),
 'pred_activity':models['recommendation'].predict(te[FULL]),
 'pred_gain':np.clip(models['gain'].predict(te[FULL]),0,None),
}

# Simulation design:
# - 10 cycles, with diminishing response factors.
# - New evidence is represented by a bounded update of the topic targeted by the activity.
# - Gain is model-predicted from the current learner state, scaled by completion and diminishing response.
# - Static keeps the initial activity; adaptive re-predicts topic/recommendation every cycle.
# - No ground-truth final grade/risk/topic/recommendation is used inside the loop.
N_CYCLES=10
DIM_FACTORS=np.array([1.00,0.72,0.50,0.40,0.32,0.27,0.23,0.20,0.18,0.16])


def simulate(strategy):
    state=te[FULL].copy()
    initial_activity=models['recommendation'].predict(state)
    rows=[]
    cum=np.zeros(len(state))
    for r in range(1,N_CYCLES+1):
        pred_grade=models['performance'].predict(state)
        pred_risk=models['risk'].predict(state)
        pred_topic=models['topic'].predict(state)
        pred_activity=models['recommendation'].predict(state)
        pred_gain=np.clip(models['gain'].predict(state),0,None)
        if strategy=='static':
            activity=initial_activity.copy()
        else:
            activity=pred_activity.copy()
        # If model emits Guided Review, route it to the currently predicted weakest topic.
        topics_for_update=[]
        effective_activity=[]
        for i,a in enumerate(activity):
            if a=='Guided Review':
                eff=TOPIC_TO_ACTIVITY.get(pred_topic[i],'Guided Review')
            else:
                eff=a
            effective_activity.append(eff)
            topics_for_update.append(ACTIVITY_TO_TOPIC.get(eff))
        effective_activity=np.array(effective_activity,dtype=object)
        # Simulated learning gain: model estimate x completion x diminishing response.
        completion=np.clip(state['recommendation_completion_rate'].to_numpy(float)/100.0,0,1)
        gain=pred_gain*completion*DIM_FACTORS[r-1]
        # Bound each update by remaining topic headroom. This prevents >100 mastery.
        deltas=np.zeros(len(state))
        for i,topic in enumerate(topics_for_update):
            if topic is None: continue
            col=TOPIC_COL[topic]
            current=float(state.iloc[i][col])
            headroom=max(0.0,100.0-current)
            delta=min(float(gain[i]),headroom)
            state.iat[i,state.columns.get_loc(col)] = current+delta
            deltas[i]=delta
            cum[i]+=delta
        for i in range(len(state)):
            rows.append({
                'student_id':int(te.iloc[i].student_id),'strategy':strategy,'round':r,
                'predicted_final_grade':float(pred_grade[i]),'predicted_risk':pred_risk[i],
                'predicted_weakest_topic':pred_topic[i],'predicted_recommendation':pred_activity[i],
                'effective_activity':effective_activity[i], 'predicted_learning_gain':float(pred_gain[i]),
                'completion_rate':float(state.iloc[i]['recommendation_completion_rate']),
                'diminishing_factor':float(DIM_FACTORS[r-1]),'simulated_gain':float(deltas[i]),
                'cumulative_simulated_gain':float(cum[i]),
                'conceptual_understanding':float(state.iloc[i]['conceptual_understanding']),
                'problem_solving':float(state.iloc[i]['problem_solving']),
                'application_skills':float(state.iloc[i]['application_skills']),
                'critical_thinking':float(state.iloc[i]['critical_thinking']),
                'advanced_topics':float(state.iloc[i]['advanced_topics']),
            })
    return pd.DataFrame(rows)

static=simulate('static')
adaptive=simulate('adaptive')
all_rounds=pd.concat([static,adaptive],ignore_index=True)
all_rounds.to_csv(OUT/'student_round_trajectories.csv',index=False)

# Summary by strategy and cycle.
summary=all_rounds.groupby(['strategy','round']).agg(
 mean_round_gain=('simulated_gain','mean'),
 median_round_gain=('simulated_gain','median'),
 sd_round_gain=('simulated_gain','std'),
 mean_cumulative_gain=('cumulative_simulated_gain','mean'),
 median_cumulative_gain=('cumulative_simulated_gain','median'),
 mean_predicted_grade=('predicted_final_grade','mean'),
 mean_predicted_learning_gain=('predicted_learning_gain','mean'),
).reset_index()
summary.to_csv(OUT/'round_summary.csv',index=False)

# End-of-cycle comparison.
end=all_rounds[all_rounds['round']==N_CYCLES]
final_summary=end.groupby('strategy').agg(
 mean_cumulative_simulated_gain=('cumulative_simulated_gain','mean'),
 median_cumulative_simulated_gain=('cumulative_simulated_gain','median'),
 sd_cumulative_simulated_gain=('cumulative_simulated_gain','std'),
 mean_predicted_final_grade=('predicted_final_grade','mean'),
 mean_predicted_learning_gain=('predicted_learning_gain','mean'),
).reset_index()
final_summary.to_csv(OUT/'static_vs_adaptive_10cycle_summary.csv',index=False)

# Student-level comparison.
piv=final_summary.copy()
st=static[static['round']==N_CYCLES][['student_id','cumulative_simulated_gain']].rename(columns={'cumulative_simulated_gain':'static_10cycle_gain'})
ad=adaptive[adaptive['round']==N_CYCLES][['student_id','cumulative_simulated_gain']].rename(columns={'cumulative_simulated_gain':'adaptive_10cycle_gain'})
student=st.merge(ad,on='student_id')
student['adaptive_minus_static']=student['adaptive_10cycle_gain']-student['static_10cycle_gain']
student.to_csv(OUT/'student_level_10cycle_comparison.csv',index=False)

# Recommendation changes for adaptive path.
a=adaptive.sort_values(['student_id','round']).copy()
a['recommendation_change']=a.groupby('student_id')['effective_activity'].transform(lambda s:s.ne(s.shift()).fillna(False))
change= a.groupby('round')['recommendation_change'].mean().reset_index(name='proportion_changed_from_previous_round')
change.loc[change['round']==1,'proportion_changed_from_previous_round']=np.nan
change.to_csv(OUT/'adaptive_recommendation_change_rates.csv',index=False)

# Activity distributions by round.
dist=adaptive.groupby(['round','effective_activity']).size().reset_index(name='count')
dist['proportion']=dist.groupby('round')['count'].transform(lambda x:x/x.sum())
dist.to_csv(OUT/'adaptive_activity_distribution.csv',index=False)

# Topic mastery trajectories.
topic_long=[]
for strategy,dat in [('static',static),('adaptive',adaptive)]:
    for r,g in dat.groupby('round'):
        for c in TOPIC_COL.values():
            topic_long.append({'strategy':strategy,'round':r,'topic':c,'mean_mastery':g[c].mean()})
pd.DataFrame(topic_long).to_csv(OUT/'topic_mastery_trajectories.csv',index=False)

# Prediction trajectories.
pred_summary=all_rounds.groupby(['strategy','round']).agg(
 mean_predicted_final_grade=('predicted_final_grade','mean'),
 mean_predicted_learning_gain=('predicted_learning_gain','mean')
).reset_index()
pred_summary.to_csv(OUT/'prediction_trajectories.csv',index=False)

# Initial-vs-final profile and model predictions.
init_df=pd.DataFrame({'student_id':te.student_id.astype(int),**{k:v for k,v in initial.items()}})
fin=adaptive[adaptive['round']==N_CYCLES][['student_id','predicted_final_grade','predicted_risk','predicted_weakest_topic','predicted_recommendation','predicted_learning_gain']].copy()
fin.columns=['student_id','final_predicted_final_grade','final_predicted_risk','final_predicted_weakest_topic','final_predicted_recommendation','final_predicted_learning_gain']
init_df.merge(fin,on='student_id').to_csv(OUT/'adaptive_initial_vs_final_predictions.csv',index=False)

meta={
 'dataset':str(DATA),'split':str(SPLIT),'seed':SEED,'train_n':len(tr),'test_n':len(te),
 'cycles':N_CYCLES,'diminishing_factors':DIM_FACTORS.tolist(),
 'models':{'performance':'Lasso','risk':'SVM class-balanced','topic':'Logistic Regression','recommendation':'Logistic Regression','learning_gain':'Lasso'},
 'simulation_note':'Proof-of-concept only. Simulated gains are generated from model-predicted expected learning gain, completion rate, diminishing response factors, and bounded topic-mastery updates. No observed longitudinal learning outcomes are used.',
 'completed':time.strftime('%Y-%m-%d %H:%M:%S')
}
(OUT/'STAGE5_METADATA.json').write_text(json.dumps(meta,indent=2))

print('STAGE 5 COMPLETE')
print('\nFINAL SUMMARY')
print(final_summary.to_string(index=False))
print('\nCHANGE RATES')
print(change.to_string(index=False))
print('\nSTUDENT DELTA DESCRIBE')
print(student['adaptive_minus_static'].describe().to_string())
