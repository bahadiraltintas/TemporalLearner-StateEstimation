from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(11,6)); ax.axis('off')
def box(x,y,w,h,text):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',linewidth=1,facecolor='white',edgecolor='black')
    ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',mutation_scale=12,linewidth=1))
box(.03,.68,.18,.16,'Latent academic\nability'); box(.03,.43,.18,.16,'Motivation /\nengagement'); box(.03,.18,.18,.16,'Participation /\nanxiety')
box(.30,.60,.20,.20,'Observed learner state\nacademic + behavioral +\nlearner-characteristic variables')
box(.30,.22,.20,.20,'Topic mastery\nM1 ... M5')
box(.62,.69,.18,.14,'Final grade')
box(.62,.48,.18,.14,'Risk label')
box(.62,.27,.18,.14,'Weakest topic\n= argmin(M1...M5)')
box(.62,.06,.18,.14,'Synthetic policy\nactivity')
box(.86,.27,.12,.20,'Synthetic\nexpected\ngain')
arrow(.21,.76,.30,.72); arrow(.21,.51,.30,.70); arrow(.21,.26,.30,.52)
arrow(.50,.70,.62,.76); arrow(.50,.65,.62,.55); arrow(.50,.30,.62,.34); arrow(.50,.26,.62,.13); arrow(.80,.13,.86,.34); arrow(.80,.34,.86,.34)
ax.text(.5,.93,'Synthetic target-generating structure (conceptual DAG)',ha='center',fontsize=13,fontweight='bold')
ax.text(.5,.01,'Arrows denote the documented synthetic construction logic; they are not causal claims about real learners.',ha='center',fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'synthetic_target_generation_dag.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/'synthetic_target_generation_dag.pdf',bbox_inches='tight'); plt.close(fig)
