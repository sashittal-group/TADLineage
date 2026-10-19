"""Label-free neighbor imputation, contiguous TAD hierarchies and RF-median EM.

Hierarchies use half-open 1Mb bin coordinates. Contiguous average-linkage
merges are always laminar. The strict-majority clades of laminar input trees
are an exact (possibly unresolved) median for interval symmetric difference.
This intentionally uses Ct=1, Cb=0, tolerance=0, not the old relaxed metric.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from numba import njit
from extract import CHROMS

HERE=Path(__file__).resolve().parent

def neighbor_weights(x,neighbors=20,self_weight=0.2,steps=1):
    d=cdist(x,x)
    np.fill_diagonal(d,np.inf)
    idx=np.argsort(d,axis=1)[:,:neighbors]
    sigma=np.maximum(np.take_along_axis(d,idx[:,-1:],axis=1),1e-12)
    w=np.zeros_like(d)
    w[np.arange(len(x))[:,None],idx]=np.exp(-np.take_along_axis(d,idx,axis=1)**2 / sigma**2)
    w/=w.sum(axis=1,keepdims=True)
    for _ in range(steps-1): w=w@w
    return self_weight*np.eye(len(x))+(1-self_weight)*w

def diffusion_weights(x,neighbors=10,self_weight=0.05,steps=5):
    # Same symmetric neighborhood geometry as spectral clustering; no assignments
    # or labels are used to construct the imputed contact matrices.
    a=kneighbors_graph(x,neighbors,include_self=True).toarray()
    a=(a+a.T)/2
    p=a/np.maximum(a.sum(axis=1,keepdims=True),1)
    w=np.linalg.matrix_power(p,steps)
    return self_weight*np.eye(len(x))+(1-self_weight)*w

@njit(cache=True)
def adjacent_tree(a,min_span=2,max_span=40):
    n=len(a)
    prefix=np.zeros((n+1,n+1))
    for i in range(n):
        for j in range(n):
            prefix[i+1,j+1]=a[i,j]+prefix[i,j+1]+prefix[i+1,j]-prefix[i,j]
    starts=np.arange(n)
    ends=np.arange(1,n+1)
    active=n
    result=np.zeros((n-1,2),np.int64)
    used=0
    while active>1:
        best=-1e300
        bestidx=0
        for t in range(active-1):
            i,k,j=starts[t],ends[t],ends[t+1]
            val=(prefix[k,j]-prefix[i,j]-prefix[k,k]+prefix[i,k])/((k-i)*(j-k))
            if val>best:
                best=val
                bestidx=t
        t=bestidx
        i,j=starts[t],ends[t+1]
        if min_span<=j-i<=max_span and j-i<n:
            result[used,0]=i
            result[used,1]=j
            used+=1
        ends[t]=j
        for v in range(t+1,active-1):
            starts[v]=starts[v+1]
            ends[v]=ends[v+1]
        active-=1
    return result[:used]

def infer_hierarchies(weights,sigma=0.7,oe=False,chroms=CHROMS):
    n=len(weights)
    cells=[[] for _ in range(n)]
    for chrom in chroms:
        raw=np.load(HERE/f'data/{chrom}_counts.npz')['counts']
        nb=raw.shape[1]
        flat=raw.reshape(n,-1).astype(float)
        flat/=np.maximum(flat.sum(axis=1,keepdims=True),1)
        pooled=(weights@flat).reshape(n,nb,nb)
        for i,m in enumerate(pooled):
            m=m+m.T
            np.fill_diagonal(m,0)
            m=gaussian_filter(m,sigma=sigma) if sigma else m
            if oe:
                # Stable distance correction estimated from the pooled map.
                expected=np.array([np.mean(np.diag(m,k)) for k in range(nb)])
                expected=np.maximum(expected,expected.max()*0.01)
                m=m/expected[np.abs(np.arange(nb)[:,None]-np.arange(nb)[None,:])]
            intervals=adjacent_tree(m)
            cells[i].extend((chrom,int(a),int(b)) for a,b in intervals)
    features=sorted(set(t for cell in cells for t in cell))
    lookup={t:j for j,t in enumerate(features)}
    row=[]; col=[]
    for i,cell in enumerate(cells):
        for t in cell:
            row.append(i); col.append(lookup[t])
    x=sparse.csr_matrix((np.ones(len(row)),(row,col)),shape=(n,len(features)))
    return x,features

def rf_distances(x,medians):
    return np.maximum(np.asarray(x.sum(axis=1))+medians.sum(axis=1)[None,:]-2*(x@medians.T),0)

def median_em(x,k=5,initial=None,seed=0,max_iter=100):
    rng=np.random.default_rng(seed)
    n,p=x.shape
    if initial is None:
        # RF kmeans++ initialization; no stage or accession order.
        chosen=[int(rng.integers(n))]
        for _ in range(k-1):
            d=rf_distances(x,x[chosen].toarray()).min(axis=1)
            d[chosen]=0
            chosen.append(int(rng.choice(n,p=d/d.sum())) if d.sum() else
                          int(rng.choice(np.setdiff1d(np.arange(n),chosen))))
        y=rf_distances(x,x[chosen].toarray()).argmin(axis=1)
    else:
        y=np.asarray(initial).copy()
    history=[]
    for iteration in range(max_iter):
        med=np.zeros((k,p))
        for c in range(k):
            members=np.where(y==c)[0]
            if len(members):
                counts=np.asarray(x[members].sum(axis=0)).ravel()
                med[c]=(counts>len(members)/2)
            else:
                # Reseed with the cell farthest from the occupied medians.
                med[c]=x[int(rng.integers(n))].toarray()[0]
        d=rf_distances(x,med)
        new=d.argmin(axis=1)
        # Preserve memberships on exact ties to avoid artificial cluster collapse.
        tied=np.isclose(d[np.arange(n),y],d.min(axis=1))
        new[tied]=y[tied]
        objective=float(d[np.arange(n),new].sum())
        history.append(objective)
        if np.array_equal(new,y): break
        y=new
    else:
        raise RuntimeError('Median EM did not converge')
    return y,med,history

def run_case(name,w,init,sigma=0.7,oe=False,restarts=6,chroms=CHROMS):
    out=HERE/'hierarchies'/name
    out.mkdir(parents=True,exist_ok=True)
    x,features=infer_hierarchies(w,sigma=sigma,oe=oe,chroms=chroms)
    sparse.save_npz(out/'trees.npz',x)
    np.save(out/'neighbor_weights.npy',w)
    (out/'intervals.json').write_text(json.dumps(features))
    solutions=[]
    for seed in range(restarts):
        y,m,h=median_em(x,initial=init if seed==0 else None,seed=seed)
        solutions.append((h[-1],y,m,h))
    _,y,m,h=min(solutions,key=lambda s:s[0])
    np.save(out/'medians.npy',m)
    np.save(out/'labels.npy',y)
    np.savez_compressed(out/'restart_labels.npz',**{f'seed{i}':s[1] for i,s in enumerate(solutions)})
    sizes=np.asarray(x.sum(axis=1)).ravel()
    d=np.maximum(sizes[:,None]+sizes[None,:]-2*(x@x.T).toarray(),0)
    np.fill_diagonal(d,0)
    sil=float(silhouette_score(d,y,metric='precomputed')) if 1<len(set(y))<len(y) else -1.
    row={'run':name,'silhouette':sil,'objective':h[-1],
         'sizes':np.bincount(y,minlength=5).tolist(),'median_sizes':m.sum(axis=1).tolist(),
         'history':h,'restart_objectives':[s[0] for s in solutions],
         'sigma':sigma,'observed_expected':oe,'restarts':restarts,'chromosomes':chroms}
    (out/'run.json').write_text(json.dumps(row,indent=2))
    print(row,flush=True)
    return row,y

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--representations',nargs='+',default=['contacts','decay_log','insulation'])
    parser.add_argument('--neighbors',nargs='+',type=int,default=[10,30,60])
    parser.add_argument('--self-weight',type=float,default=0.2)
    parser.add_argument('--steps',type=int,default=1)
    parser.add_argument('--dimensions',type=int,default=10)
    parser.add_argument('--oe',action='store_true')
    parser.add_argument('--baselines',action='store_true')
    parser.add_argument('--sigma',type=float,default=0.7)
    parser.add_argument('--graph',choices=['directed','symmetric'],default='directed')
    parser.add_argument('--init',choices=['kmeans','spectral'],default='kmeans')
    args=parser.parse_args()
    reps=np.load(HERE/'signals/representations.npz')
    n=len(reps['contacts'])
    configs=[]
    if args.baselines:
        configs.extend([('raw',np.eye(n),None),('global_pool',0.2*np.eye(n)+0.8*np.ones((n,n))/n,None)])
    for rep in args.representations:
        x=reps[rep][:,:args.dimensions]
        for nn in args.neighbors:
            if args.init=='spectral':
                init=SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=nn,
                                       assign_labels='cluster_qr',random_state=0).fit_predict(x)
            else:
                init=KMeans(5,n_init=30,random_state=0).fit_predict(x)
            weight_fn=diffusion_weights if args.graph=='symmetric' else neighbor_weights
            w=weight_fn(x,nn,args.self_weight,args.steps)
            name=f'{rep}_d{args.dimensions}_nn{nn}_self{args.self_weight}_steps{args.steps}_sigma{args.sigma}_oe{int(args.oe)}'
            if args.graph!='directed' or args.init!='kmeans':
                name+=f'_{args.graph}_{args.init}'
            configs.append((name,w,init))
    out=HERE/'hierarchies'
    out.mkdir(exist_ok=True)
    for name,w,init in configs:
        if not (out/name/'run.json').exists():
            run_case(name,w,init,sigma=args.sigma,oe=args.oe)
    rows=[]; pred={}
    for path in sorted(out.glob('*/run.json')):
        r=json.loads(path.read_text())
        rows.append(r)
        pred[r['run']]=np.load(path.parent/'labels.npy')
    (out/'runs.json').write_text(json.dumps(rows,indent=2))
    np.savez_compressed(out/'predictions.npz',**pred)
    (out/'selected.json').write_text(json.dumps(max(rows,key=lambda r:r['silhouette']),indent=2))

if __name__=='__main__': main()
