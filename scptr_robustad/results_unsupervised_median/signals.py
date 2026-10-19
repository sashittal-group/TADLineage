"""Exploratory label-free representations; stage labels never enter this module."""
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import pdist, squareform

HERE=Path(__file__).resolve().parent

def pca(x,n=20,scale=False):
    if scale:
        x=StandardScaler().fit_transform(x)
    return PCA(n_components=min(n,x.shape[1],len(x)-1),random_state=0).fit_transform(x)

def normalize(x):
    return x / np.maximum(x.sum(axis=1,keepdims=True),1)

def representations():
    f=np.load(HERE/'data/features.npz')
    decay=f['decay'].sum(axis=1)
    prob=normalize(decay)
    reps={}
    reps['decay_sqrt']=np.sqrt(prob)
    reps['decay_log']=pca(np.log(prob+1e-5),10)
    reps['decay_scaled']=pca(np.log(prob+1e-5),10,True)
    for name in ('coverage','insulation','contacts'):
        x=f[name]
        x=np.sqrt(normalize(x))
        reps[name]=pca(x,30)
    chromdecay=np.sqrt(normalize(f['decay'].reshape(len(prob),-1)))
    reps['chrom_decay']=pca(chromdecay,20)
    for other in ('insulation','coverage','contacts'):
        a=reps['decay_log'][:,:5]
        b=reps[other][:,:10]
        reps['decay_'+other]=pca(np.column_stack((a/np.linalg.norm(a),b/np.linalg.norm(b))),15)
    return reps

def main():
    out=HERE/'signals'
    out.mkdir(exist_ok=True)
    reps=representations()
    np.savez_compressed(out/'representations.npz',**reps)
    rows=[]
    predictions={}
    for name,xfull in reps.items():
        for dim in (3,5,10,20):
            if dim>xfull.shape[1]: continue
            x=xfull[:,:dim] if name!='decay_sqrt' else pca(xfull,dim)
            for method in ('kmeans','gmm_diag','gmm_full','spectral'):
                key=f'{name}__d{dim}__{method}'
                if method=='kmeans':
                    est=KMeans(5,n_init=30,random_state=0)
                elif method.startswith('gmm'):
                    est=GaussianMixture(5,covariance_type=method.split('_')[1],n_init=5,
                                        reg_covar=1e-6,random_state=0)
                else:
                    est=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=20,
                                            assign_labels='cluster_qr',random_state=0)
                y=est.fit_predict(x)
                sil=float(silhouette_score(x,y))
                row={'run':key,'representation':name,'dimensions':dim,'method':method,
                     'silhouette':sil,'sizes':np.bincount(y,minlength=5).tolist()}
                rows.append(row)
                predictions[key]=y
            print(name,dim,flush=True)
    (out/'runs.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(out/'predictions.npz',**predictions)
    selected=max(rows,key=lambda r:r['silhouette'])
    (out/'selected.json').write_text(json.dumps(selected,indent=2))
    print('Silhouette selection:',selected,flush=True)

if __name__=='__main__': main()
