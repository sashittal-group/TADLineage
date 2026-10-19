"""Reproduce the frozen stage-blind experiment; no stage metadata is read.

Default k=5 is a benchmark assumption. --k auto uses RF silhouette over 2..9.
All generated files stay inside this experiment directory.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from signals import pca,normalize
from hierarchy import diffusion_weights,infer_hierarchies,median_em
from extract import HERE,CHROMS,SIZES

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--k',default='5')
    parser.add_argument('--neighbors',type=int,default=10)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--self-weight',type=float,default=.05)
    parser.add_argument('--restarts',type=int,default=20)
    parser.add_argument('--output',default='final')
    args=parser.parse_args()
    out=(HERE/args.output).resolve()
    if not out.is_relative_to(HERE) or out==HERE:
        raise ValueError('Output must be a subdirectory of this experiment folder')
    out.mkdir(parents=True,exist_ok=True)
    if not (HERE/'data/features.npz').exists():
        from extract import main as extract_main
        extract_main()
    ids=json.loads((HERE/'data/cells.json').read_text())
    if not (2<=args.neighbors<len(ids) and 0<=args.self_weight<=1 and args.steps>=1 and args.restarts>=1):
        raise ValueError('Invalid neighbor count, self weight, step count or restart count')
    if args.k!='auto' and not 2<=int(args.k)<len(ids):
        raise ValueError('k must be auto or an integer in [2, n_cells-1]')
    f=np.load(HERE/'data/features.npz')
    x=pca(np.sqrt(normalize(f['contacts'])),30)[:,:10]
    w=diffusion_weights(x,args.neighbors,args.self_weight,args.steps)
    trees,intervals=infer_hierarchies(w,sigma=.7)
    size=np.asarray(trees.sum(axis=1)).ravel()
    d=np.maximum(size[:,None]+size[None,:]-2*(trees@trees.T).toarray(),0)
    np.fill_diagonal(d,0)
    candidates=range(2,10) if args.k=='auto' else [int(args.k)]
    runs=[]
    for k in candidates:
        init=SpectralClustering(k,affinity='nearest_neighbors',n_neighbors=args.neighbors,
                                assign_labels='cluster_qr',random_state=0).fit_predict(x)
        solutions=[]
        for seed in range(args.restarts):
            y,m,h=median_em(trees,k,initial=init if seed==0 else None,seed=seed)
            solutions.append((h[-1],y,m,h))
        objective,y,m,h=min(solutions,key=lambda s:s[0])
        silhouette=float(silhouette_score(d,y,metric='precomputed'))
        runs.append((silhouette,k,objective,y,m,h,solutions))
    silhouette,k,objective,y,m,h,solutions=max(runs,key=lambda r:r[0])
    metadata={'n_cells':len(ids),'k':k,'k_selection':'RF silhouette over 2..9' if args.k=='auto' else 'fixed benchmark k',
              'neighbors':args.neighbors,'diffusion_steps':args.steps,'self_weight':args.self_weight,
              'pca_dimensions':10,'pca_fitted_components':30,'resolution_bp':1000000,
              'input_resolution_bp':250000,'smoothing_sigma_bins':.7,'min_domain_bins':2,'max_domain_bins':40,
              'chromosomes':CHROMS,'distance':'interval RF; Ct=1, Cb=0, tolerance=0',
              'imputation':f'{args.self_weight} I + {1-args.self_weight} P^{args.steps}',
              'stage_labels_used_in_fit':False,'reference_panels_used':False,
              'objective':objective,'silhouette':silhouette,'history':h,
              'cluster_sizes':np.bincount(y,minlength=k).tolist(),
              'median_sizes':m.sum(axis=1).astype(int).tolist(),
              'restart_objectives':[s[0] for s in solutions],
              'model_selection':[{'k':r[1],'silhouette':r[0],'objective':r[2]} for r in runs],
              'development_status':'Exploratory; stage ARI was inspected during method development.'}
    (out/'run.json').write_text(json.dumps(metadata,indent=2))
    np.savez_compressed(out/'fit.npz',labels=y,medians=m,embedding=x,weights=w,
                         restart_labels=np.array([s[1] for s in solutions]))
    sparse.save_npz(out/'trees.npz',trees)
    (out/'intervals.json').write_text(json.dumps(intervals))
    (out/'assignments.tsv').write_text('cell_id\tcluster\n'+''.join(f'{s}\t{int(c)}\n' for s,c in zip(ids,y)))
    with gzip.open(out/'imputed_hierarchies.tsv.gz','wt') as handle:
        handle.write('cell_id\tchromosome\tstart_bp\tend_bp\n')
        for i,cell in enumerate(ids):
            for j in trees[i].indices:
                chrom,a,b=intervals[j]
                end=min(b*1000000,SIZES[CHROMS.index(chrom)])
                handle.write(f'{cell}\t{chrom}\t{a*1000000}\t{end}\n')
    with (out/'median_hierarchies.tsv').open('w') as handle:
        handle.write('cluster\tchromosome\tstart_bp\tend_bp\n')
        for c in range(k):
            for j in np.flatnonzero(m[c]):
                chrom,a,b=intervals[j]
                end=min(b*1000000,SIZES[CHROMS.index(chrom)])
                handle.write(f'{c}\t{chrom}\t{a*1000000}\t{end}\n')
    print(json.dumps(metadata,indent=2),flush=True)

if __name__=='__main__': main()
