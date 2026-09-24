from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_frozen_v1" / "synthetic_learner_state_dataset_v1.0.csv"
RESULTS = ROOT / "results"
SEED = 42
TEST_SIZE = 0.20

ACADEMIC = [
    "age","gender","class_year","previous_gpa","prior_course_grade",
    "attendance_rate","absence_hours","weekly_questions","quiz_average",
    "assignment_average","assignment_on_time","midterm","practice_test_score"
]
BEHAVIORAL = [
    "study_hours_week","lms_login_count","video_watch_minutes",
    "time_management","self_regulation","recommendation_completion_rate"
]
PSYCHOLOGICAL = [
    "motivation","self_efficacy","academic_anxiety","preferred_learning_type"
]
TOPIC = [
    "conceptual_understanding","problem_solving","application_skills",
    "critical_thinking","advanced_topics"
]
FULL = ACADEMIC + BEHAVIORAL + PSYCHOLOGICAL + TOPIC

TARGETS = {
    "performance": "final_grade",
    "risk": "learning_risk_level",
    "topic": "weakest_topic",
    "recommendation": "recommended_activity",
    "gain": "expected_learning_gain",
}

ACTIVITIES = [
    "Concept Review",
    "Practice Questions",
    "Application Exercise",
    "Critical Thinking Activity",
    "Advanced Practice",
]
TOPICS = [
    "Conceptual Understanding",
    "Problem Solving",
    "Application Skills",
    "Critical Thinking",
    "Advanced Topics",
]
TOPIC_TO_ACTIVITY = dict(zip(TOPICS, ACTIVITIES))
ACTIVITY_TO_TOPIC = {v:k for k,v in TOPIC_TO_ACTIVITY.items()}
TOPIC_COL = {
    "Conceptual Understanding":"conceptual_understanding",
    "Problem Solving":"problem_solving",
    "Application Skills":"application_skills",
    "Critical Thinking":"critical_thinking",
    "Advanced Topics":"advanced_topics",
}
