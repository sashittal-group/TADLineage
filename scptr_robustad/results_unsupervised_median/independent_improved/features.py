"""Single-chromosome representations. No labels or other-chromosome features."""
from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from signals import pca
from extract import CHROMS

def probability(x):
    return x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)

def contact_embedding(m, transform='sqrt'):
    a,b = np.triu_indices(m.shape[1], 1)
    x = probability(m[:,a,b])
    if transform == 'sqrt': x = np.sqrt(x)
    elif transform == 'log': x = np.log1p(x * x.shape[1])
    return pca(x,30)

def build_features(counts, decay, edges):
    """Only numeric inputs from one chromosome; no identifiers are arguments."""
    n,nb,_ = counts.shape
    sym = counts + counts.transpose(0,2,1)
    sym[:,np.arange(nb),np.arange(nb)] = 0
    reps = {'raw': contact_embedding(sym)}
    smooth = gaussian_filter(sym, (0,.7,.7))
    reps['smooth'] = contact_embedding(smooth)
    reps['smooth_log'] = contact_embedding(smooth,'log')
    # Estimate a shared distance background from this chromosome alone.
    mean = smooth.mean(axis=0)
    expected = np.array([np.diag(mean,k).mean() for k in range(nb)])
    expected = np.maximum(expected, expected.max()*.005)
    distance = np.abs(np.arange(nb)[:,None]-np.arange(nb)[None,:])
    reps['distance_corrected'] = contact_embedding(smooth/np.sqrt(expected[distance]))
    degree = smooth.sum(axis=2)
    balanced = smooth / np.maximum(np.sqrt(degree[:,:,None]*degree[:,None,:]),1e-12)
    reps['balanced'] = contact_embedding(balanced)
    # Restart diffusion within each cell's genomic contact map (not between cells).
    transition = smooth / np.maximum(smooth.sum(axis=2,keepdims=True),1e-12)
    walk = .5*transition
    power = transition.copy()
    for step in range(2,5):
        power = power @ transition
        walk += (.5**step)*power
    walk = (walk+walk.transpose(0,2,1))/2
    reps['walk'] = contact_embedding(walk)
    reps['walk_log'] = contact_embedding(walk,'log')
    a,b = np.triu_indices(nb,1)
    upper = walk[:,a,b]
    threshold = np.quantile(upper,.8,axis=1,keepdims=True)
    reps['walk_binary'] = pca((upper>threshold).astype(float),30)
    valid = edges[:-1]>=20000
    prob = probability(decay[:,valid]+.5)
    reps['decay'] = pca(np.log(prob),20)
    ins = []
    for window in (2,5,10):
        values=np.zeros((n,nb))
        for b in range(1,nb):
            values[:,b]=smooth[:,max(0,b-window):b,b:min(nb,b+window)].sum(axis=(1,2))
        ins.append(np.sqrt(probability(values)))
    reps['insulation'] = pca(np.concatenate(ins,axis=1),30)
    for name in ('raw','smooth','walk'):
        x=reps[name][:,:10]; z=reps['decay'][:,:5]
        reps[name+'_decay']=pca(np.column_stack((x/np.linalg.norm(x),z/np.linalg.norm(z))),15)
    return reps

def main():
    out=HERE/'features'; out.mkdir(exist_ok=True)
    # This existing cache stores a separate histogram for every chromosome.
    # Slice before feature construction; genome-wide hist/count fields are unused.
    fine=np.load(PARENT/'data/raw_profiles.npz')
    for ci,chrom in enumerate(CHROMS):
        target=out/f'{chrom}.npz'
        if target.exists(): continue
        counts=np.load(PARENT/f'data/{chrom}_counts.npz')['counts']
        decay=fine['chrom_hist'][:,ci,:]
        np.savez_compressed(out/f'{chrom}_decay.npz',hist=decay,edges=fine['edges'])
        reps=build_features(counts,decay,fine['edges'])
        np.savez_compressed(target,**reps)
        print(chrom,list(reps),flush=True)

if __name__=='__main__': main()
