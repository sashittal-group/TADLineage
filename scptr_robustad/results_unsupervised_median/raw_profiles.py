"""Extract fine-scale contact-distance signal omitted by the 250kb matrices."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
EDGES=np.r_[0,np.geomspace(1000,250_000_000,101)]

def extract(item):
    cell,path=item
    out=HERE/'data/raw_profiles'
    target=out/f'{cell}.npz'
    if target.exists(): return
    hist=np.zeros(len(EDGES)-1)
    chrom_hist=np.zeros((20,len(EDGES)-1))
    counts=np.zeros(3)
    chroms=[f'chr{i}' for i in range(1,20)]+['chrX']
    for frame in pd.read_csv(path,sep='\t',header=None,usecols=[1,2,4,5],
                             names=['c1','p1','c2','p2'],chunksize=500000):
        cis=(frame.c1==frame.c2)&frame.c1.isin(chroms)
        counts[0]+=len(frame)
        counts[1]+=cis.sum()
        a=frame.loc[cis]
        distances=np.abs(a.p1.to_numpy()-a.p2.to_numpy())
        hist+=np.histogram(distances,bins=EDGES)[0]
        c=a.c1.to_numpy()
        for j,chrom in enumerate(chroms):
            chrom_hist[j]+=np.histogram(distances[c==chrom],bins=EDGES)[0]
    np.savez_compressed(target,hist=hist,chrom_hist=chrom_hist,counts=counts,edges=EDGES)

def main():
    out=HERE/'data/raw_profiles'
    out.mkdir(exist_ok=True)
    ids=json.loads((HERE/'data/cells.json').read_text())
    paths={p.name.split('_')[0]:p for p in (ROOT/'scptr/cell_differentiation').glob('*/*.txt')}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,_ in enumerate(pool.map(extract,[(s,paths[s]) for s in ids])):
            if i%25==0: print(i+1,len(ids),flush=True)
    blobs=[np.load(out/f'{s}.npz') for s in ids]
    np.savez_compressed(HERE/'data/raw_profiles.npz',
                         hist=np.array([b['hist'] for b in blobs]),
                         chrom_hist=np.array([b['chrom_hist'] for b in blobs]),
                         counts=np.array([b['counts'] for b in blobs]),edges=EDGES)

if __name__=='__main__': main()
