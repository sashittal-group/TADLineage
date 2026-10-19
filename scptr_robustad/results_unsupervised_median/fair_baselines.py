"""Give ablations the same spectral start plus 19 random starts as the final fit."""
import json
import shutil
import numpy as np
from scipy import sparse
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from hierarchy import HERE,median_em

def main():
    x=np.load(HERE/'signals/representations.npz')['contacts'][:,:10]
    init=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=10,
                            assign_labels='cluster_qr',random_state=0).fit_predict(x)
    for name in ('raw','global_pool','control_shuffled_neighbors'):
        out=HERE/'hierarchies'/name
        archive=out/'initial_6_restart'
        if not archive.exists():
            archive.mkdir()
            for f in ('run.json','labels.npy','medians.npy','restart_labels.npz'):
                shutil.copy2(out/f,archive/f)
        initial=init
        if name=='control_shuffled_neighbors':
            perm=np.random.default_rng(2026).permutation(len(x))
            initial=init[perm]
        t=sparse.load_npz(out/'trees.npz')
        solutions=[]
        for seed in range(20):
            y,m,h=median_em(t,initial=initial if seed==0 else None,seed=seed)
            solutions.append((h[-1],y,m,h))
        objective,y,m,h=min(solutions,key=lambda s:s[0])
        np.save(out/'labels.npy',y); np.save(out/'medians.npy',m)
        np.savez_compressed(out/'restart_labels.npz',**{f'seed{i}':s[1] for i,s in enumerate(solutions)})
        sizes=np.asarray(t.sum(axis=1)).ravel()
        d=np.maximum(sizes[:,None]+sizes[None,:]-2*(t@t.T).toarray(),0)
        np.fill_diagonal(d,0)
        r=json.loads((out/'run.json').read_text())
        r.update(objective=objective,silhouette=float(silhouette_score(d,y,metric='precomputed')),
                 sizes=np.bincount(y,minlength=5).tolist(),median_sizes=m.sum(axis=1).tolist(),
                 history=h,restarts=20,restart_objectives=[s[0] for s in solutions],
                 initialization='spectral + 19 random RF kmeans++ starts')
        (out/'run.json').write_text(json.dumps(r,indent=2))
        print(name,objective,flush=True)

if __name__=='__main__': main()
