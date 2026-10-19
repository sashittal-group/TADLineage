"""Fit chromosome-local imputation/median candidates without reading labels.

Candidate settings may come from the explicitly ARI-tuned exploratory screen;
that provenance is recorded. No known-label grouping or cross-chromosome graph
is used by any candidate, irrespective of how its settings were selected.
"""
import argparse
import json
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from features import HERE,PARENT
from hierarchy import diffusion_weights,infer_hierarchies,median_em

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',default='candidate_plan.json')
    args=parser.parse_args()
    plan=json.loads((HERE/args.plan).read_text())
    for spec in plan:
        chrom=spec['chromosome']
        reps=np.load(HERE/f'features/{chrom}.npz')
        x=reps[spec['representation']][:,:spec['dimensions']]
        init=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=spec['neighbors'],
                                assign_labels='cluster_qr',random_state=0).fit_predict(x)
        settings=spec.get('imputation_settings',[(1,.05),(3,.05),(5,.05),(3,.2)])
        for steps,self_weight in settings:
            name=f'{spec["run"]}_t{steps}_self{self_weight}'
            out=HERE/'fits'/chrom/name
            if (out/'run.json').exists(): continue
            out.mkdir(parents=True,exist_ok=True)
            w=diffusion_weights(x,spec['neighbors'],self_weight,steps)
            trees,intervals=infer_hierarchies(w,sigma=.7,chroms=[chrom])
            fits=[]
            for seed in range(20):
                y,m,h=median_em(trees,k=5,initial=init if seed==0 else None,seed=seed)
                assert np.all(np.diff(h)<=1e-8)
                fits.append((h[-1],y,m,h))
            objective,y,m,h=min(fits,key=lambda s:s[0])
            for c in range(5):
                if np.any(y==c):
                    assert np.array_equal(m[c],np.asarray(trees[y==c].sum(axis=0)).ravel()>np.sum(y==c)/2)
            size=np.asarray(trees.sum(axis=1)).ravel()
            d=np.maximum(size[:,None]+size[None,:]-2*(trees@trees.T).toarray(),0)
            np.fill_diagonal(d,0)
            sil=float(silhouette_score(d,y,metric='precomputed')) if len(set(y))>1 else -1.
            row={**spec,'run':name,'diffusion_steps':steps,'self_weight':self_weight,
                 'n_cells':len(y),'k':5,'restarts':20,'objective':objective,'history':h,
                 'silhouette':sil,'sizes':np.bincount(y,minlength=5).tolist(),
                 'restart_objectives':[r[0] for r in fits],
                 'source_chromosomes':[chrom],'stage_labels_used_in_fit':False,
                 'resolution_bp':1000000,'sigma':.7}
            sparse.save_npz(out/'trees.npz',trees)
            np.savez_compressed(out/'fit.npz',labels=y,medians=m,
                                 restart_labels=np.array([r[1] for r in fits]))
            (out/'intervals.json').write_text(json.dumps(intervals))
            (out/'run.json').write_text(json.dumps(row,indent=2))
            print(chrom,name,'objective',objective,flush=True)

if __name__=='__main__': main()
