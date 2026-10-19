"""Second exploratory feature screen. Selection uses silhouette, never labels."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from signals import pca,normalize

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'signals_refined'; out.mkdir(exist_ok=True)
    base=dict(np.load(HERE/'signals/representations.npz'))
    f=np.load(HERE/'data/raw_profiles.npz')
    reps={}
    for cut in (1000,20000,100000,500000):
        x=f['hist'][:,f['edges'][:-1]>=cut]
        prob=normalize(x)
        reps[f'raw_sqrt_{cut}']=pca(np.sqrt(prob),20)
        reps[f'raw_log_{cut}']=pca(np.log(prob+1e-5),20)
        reps[f'raw_scaled_{cut}']=pca(np.log(prob+1e-5),20,True)
    for dim in (3,5,10):
        a=base['contacts'][:,:dim]
        b=reps['raw_log_20000'][:,:dim]
        for weight in (0.25,0.5,1,2):
            reps[f'contact_raw_d{dim}_w{weight}']=pca(np.column_stack((a/np.linalg.norm(a),weight*b/np.linalg.norm(b))),2*dim)
    reps['contacts']=base['contacts']
    reps['insulation']=base['insulation']
    np.savez_compressed(out/'representations.npz',**reps)
    rows=[]; pred={}
    for name,full in reps.items():
        for dim in (3,5,10,20):
            if dim>full.shape[1]: continue
            x=full[:,:dim]
            for method in ('kmeans','gmm_full','gmm_diag','spectral10','spectral30','spectral60'):
                if method=='kmeans': est=KMeans(5,n_init=30,random_state=0)
                elif method.startswith('gmm'):
                    est=GaussianMixture(5,covariance_type=method.split('_')[1],n_init=5,
                                        reg_covar=1e-6,random_state=0)
                else:
                    est=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=int(method[8:]),
                                          assign_labels='cluster_qr',random_state=0)
                y=est.fit_predict(x)
                key=f'{name}__d{dim}__{method}'
                row={'run':key,'representation':name,'dimensions':dim,'method':method,
                     'silhouette':float(silhouette_score(x,y)),
                     'sizes':np.bincount(y,minlength=5).tolist()}
                rows.append(row); pred[key]=y
        print(name,flush=True)
    (out/'runs.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(out/'predictions.npz',**pred)
    (out/'selected.json').write_text(json.dumps(max(rows,key=lambda r:r['silhouette']),indent=2))
    # Fit scripts see only these numeric representations, never evaluation output.
    base.update(reps)
    np.savez_compressed(HERE/'signals/representations.npz',**base)

if __name__=='__main__': main()
