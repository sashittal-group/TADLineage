"""Post-hoc screen evaluation and explicit exploratory candidate selection.

The shortlist uses ARI on these same cells. It is benchmark tuning, not blind
model selection. An additional silhouette-selected graph is retained separately.
"""
import json
import numpy as np
from sklearn.metrics import adjusted_rand_score
from features import HERE,PARENT,CHROMS
from evaluate import truth

def main():
    y=truth(); plan=[]; summary=[]
    for chrom in CHROMS:
        out=HERE/'screen'/chrom
        if not (out/'runs.json').exists(): continue
        rows=json.loads((out/'runs.json').read_text())
        pred=np.load(out/'predictions.npz')
        for r in rows: r['ARI']=float(adjusted_rand_score(y,pred[r['run']]))
        (out/'evaluation.json').write_text(json.dumps(rows,indent=2))
        ranked=sorted(rows,key=lambda r:r['ARI'],reverse=True)
        chosen=[]; reps=set()
        for row in ranked:
            if row['representation'] not in reps:
                chosen.append(row); reps.add(row['representation'])
            if len(chosen)==3: break
        blind=max(rows,key=lambda r:r['silhouette'])
        for row in chosen:
            plan.append({'chromosome':chrom,**{k:row[k] for k in ('run','representation','dimensions','neighbors')},
                         'selection':'exploratory_same-cell_ARI'})
        if blind['run'] not in [r['run'] for r in chosen]:
            plan.append({'chromosome':chrom,**{k:blind[k] for k in ('run','representation','dimensions','neighbors')},
                         'selection':'silhouette_no_stage_labels'})
        summary.append({'chromosome':chrom,'best':ranked[0],'silhouette_selected':blind})
        print(chrom,ranked[0]['run'],round(ranked[0]['ARI'],4),'blind',round(blind['ARI'],4),flush=True)
    (HERE/'candidate_plan.json').write_text(json.dumps(plan,indent=2))
    (HERE/'screen_summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
