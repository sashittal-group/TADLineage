"""Check two hash-selected cells' cached matrices against their raw read pairs."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from extract import HERE,ROOT,SOURCE,CHROMS,SIZES

def main():
    ids=json.loads((HERE/'data/cells.json').read_text())[:2]
    paths={p.name.split('_')[0]:p for p in (ROOT/'scptr/cell_differentiation').glob('*/*.txt')}
    results=[]
    for cell in ids:
        counts={c:Counter() for c in CHROMS}
        for frame in pd.read_csv(paths[cell],sep='\t',header=None,usecols=[1,2,4,5],
                                 names=['c1','p1','c2','p2'],chunksize=500000):
            for chrom,size in zip(CHROMS,SIZES):
                s=frame[(frame.c1==chrom)&(frame.c2==chrom)]
                a=np.minimum(s.p1.to_numpy()//250000,size//250000)
                b=np.minimum(s.p2.to_numpy()//250000,size//250000)
                counts[chrom].update((int(min(i,j)),int(max(i,j))) for i,j in zip(a,b) if i!=j)
        for chrom in CHROMS:
            name=paths[cell].stem
            source=SOURCE/chrom/'cell_matrix_mm10'/f'{name}_{chrom}.txt'
            arr=np.loadtxt(source,dtype=int).reshape(-1,3)
            observed={(int(a),int(b)):int(c) for a,b,c in arr}
            if observed!=dict(counts[chrom]):
                raise AssertionError(f'Raw count mismatch: {cell} {chrom}')
        results.append({'cell':cell,'chromosomes_checked':len(CHROMS),'exact_match':True})
    (HERE/'data/raw_count_verification.json').write_text(json.dumps(results,indent=2))
    print(results)

if __name__=='__main__': main()
