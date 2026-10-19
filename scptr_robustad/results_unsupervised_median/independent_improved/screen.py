"""A fixed spectral graph screen; no stage labels are loaded for fitting."""
import json
import argparse
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from features import HERE,CHROMS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chromosomes',nargs='+',default=CHROMS)
    args=parser.parse_args()
    for chrom in args.chromosomes:
        out=HERE/'screen'/chrom; out.mkdir(parents=True,exist_ok=True)
        if (out/'runs.json').exists(): continue
        reps=np.load(HERE/f'features/{chrom}.npz')
        rows=[]; pred={}
        for name in reps.files:
            for dim in (5,10,20):
                if dim>reps[name].shape[1]: continue
                x=reps[name][:,:dim]
                for nn in (6,10,20,40):
                    key=f'{name}_d{dim}_nn{nn}'
                    y=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=nn,
                                         assign_labels='cluster_qr',random_state=0).fit_predict(x)
                    pred[key]=y
                    rows.append(dict(run=key,representation=name,dimensions=dim,neighbors=nn,
                                     silhouette=float(silhouette_score(x,y)),
                                     sizes=np.bincount(y,minlength=5).tolist()))
        np.savez_compressed(out/'predictions.npz',**pred)
        (out/'runs.json').write_text(json.dumps(rows,indent=2))
        (out/'selected_without_labels.json').write_text(json.dumps(max(rows,key=lambda r:r['silhouette']),indent=2))
        print(chrom,len(rows),'completed',flush=True)

if __name__=='__main__': main()
