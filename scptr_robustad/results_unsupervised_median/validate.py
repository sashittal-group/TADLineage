"""Freeze a successful exploratory configuration and examine robustness.

All fitting/model selection here is label-free. Evaluation runs separately.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from hierarchy import diffusion_weights,run_case,median_em

HERE=Path(__file__).resolve().parent
REFERENCE='contacts_d10_nn10_self0.05_steps5_sigma0.7_oe0_symmetric_spectral'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sensitivity',action='store_true')
    args=parser.parse_args()
    x=np.load(HERE/'signals/representations.npz')['contacts'][:,:10]
    if args.sensitivity:
        for steps in (3,5,8):
            for nn in (8,10,12):
                name=f'contacts_d10_nn{nn}_self0.05_steps{steps}_sigma0.7_oe0_symmetric_spectral'
                if (HERE/'hierarchies'/name/'run.json').exists(): continue
                init=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=nn,
                                        assign_labels='cluster_qr',random_state=0).fit_predict(x)
                run_case(name,diffusion_weights(x,nn,.05,steps),init)
        return
    folder=HERE/'validation'; folder.mkdir(exist_ok=True)
    trees=sparse.load_npz(HERE/'hierarchies'/REFERENCE/'trees.npz')
    sizes=np.asarray(trees.sum(axis=1)).ravel()
    d=np.maximum(sizes[:,None]+sizes[None,:]-2*(trees@trees.T).toarray(),0)
    np.fill_diagonal(d,0)
    rows=[]; preds={}
    for k in range(2,10):
        init=SpectralClustering(k,affinity='nearest_neighbors',n_neighbors=10,
                                assign_labels='cluster_qr',random_state=0).fit_predict(x)
        runs=[]
        for seed in range(20):
            y,m,h=median_em(trees,k,initial=init if seed==0 else None,seed=seed)
            runs.append((h[-1],y,m,h))
            if k==5:
                preds[f'seed{seed}']=y
                rows.append({'run':f'seed{seed}','k':k,'objective':h[-1],
                             'silhouette':float(silhouette_score(d,y,metric='precomputed')),
                             'sizes':np.bincount(y,minlength=k).tolist()})
        _,y,m,h=min(runs,key=lambda s:s[0])
        row={'run':f'k{k}','k':k,'objective':h[-1],
             'silhouette':float(silhouette_score(d,y,metric='precomputed')),
             'sizes':np.bincount(y,minlength=k).tolist()}
        rows.append(row); preds[f'k{k}']=y
        print(row,flush=True)
    (folder/'runs.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(folder/'predictions.npz',**preds)
    selected=max([r for r in rows if r['run'].startswith('k')],key=lambda r:r['silhouette'])
    (folder/'selected.json').write_text(json.dumps(selected,indent=2))

if __name__=='__main__': main()
