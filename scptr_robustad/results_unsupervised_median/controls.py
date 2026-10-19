"""Shuffled-neighbor negative control and chromosome-disjoint validation."""
import numpy as np
from sklearn.cluster import SpectralClustering
from hierarchy import run_case,diffusion_weights
from signals import pca,normalize
from extract import CHROMS,HERE

def fit_initial(x):
    return SpectralClustering(5,affinity='nearest_neighbors',n_neighbors=10,
                             assign_labels='cluster_qr',random_state=0).fit_predict(x)

def main():
    x=np.load(HERE/'signals/representations.npz')['contacts'][:,:10]
    w=diffusion_weights(x,10,.05,5)
    perm=np.random.default_rng(2026).permutation(len(x))
    run_case('control_shuffled_neighbors',w[perm][:,perm],fit_initial(x)[perm])
    for tag,graph_chroms,tad_chroms in (
            ('odd_to_even',CHROMS[::2],CHROMS[1::2]),
            ('even_to_odd',CHROMS[1::2],CHROMS[::2])):
        arrays=[]
        for chrom in graph_chroms:
            m=np.load(HERE/f'data/{chrom}_counts.npz')['counts']
            a,b=np.triu_indices(m.shape[1],1)
            arrays.append(m[:,a,b])
        x=pca(np.sqrt(normalize(np.concatenate(arrays,axis=1))),10)
        run_case(f'validation_{tag}',diffusion_weights(x,10,.05,5),
                 fit_initial(x),chroms=tad_chroms)

if __name__=='__main__': main()
