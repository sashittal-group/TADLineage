"""Read label-free mm10 contact counts; write every artifact beside this script."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'schicptr_scptr/work/250kb'
CHROMS = [f'chr{i}' for i in range(1, 20)] + ['chrX']
SIZES = [195471971,182113224,160039680,156508116,151834684,149736546,
         145441459,129401213,124595110,130694993,122082543,120129022,
         120421639,124902244,104043685,98207768,94987271,90702639,
         61431566,171031299]

def main():
    out = HERE / 'data'
    out.mkdir(exist_ok=True)
    files = list((SOURCE / 'chr1/cell_matrix_mm10').glob('*.txt'))
    # Hash order prevents accession order or stage-bearing filenames from being features.
    names = sorted([p.name.removesuffix('_chr1.txt') for p in files],
                   key=lambda s: hashlib.sha256(s.split('_')[0].encode()).hexdigest())
    ids = [s.split('_')[0] for s in names]
    (out / 'cells.json').write_text(json.dumps(ids, indent=2))
    n = len(names)
    edges = np.unique(np.rint(np.geomspace(1, 800, 65)).astype(int))
    decay = np.zeros((n,20,len(edges)-1), np.float64)
    coverage, local, contacts = [], [], []
    manifest = []
    for c, (chrom,size) in enumerate(zip(CHROMS,SIZES)):
        nb = (size + 999999)//1000000
        mats = np.zeros((n,nb,nb),np.float32)
        for i,name in enumerate(names):
            path = SOURCE / chrom / 'cell_matrix_mm10' / f'{name}_{chrom}.txt'
            raw = path.read_bytes()
            arr = np.fromstring(raw.decode(),sep=' ').reshape(-1,3)
            a,b = arr[:,:2].astype(int).T
            w = arr[:,2]
            decay[i,c] = np.histogram(b-a,bins=edges,weights=w)[0]
            np.add.at(mats[i],(a//4,b//4),w)
            manifest.append({'cell':ids[i], 'chrom':chrom, 'path':str(path.relative_to(ROOT)),
                             'sha256':hashlib.sha256(raw).hexdigest()})
        # Original files contain upper triangle, with 250kb diagonal omitted.
        # 1Mb diagonal includes contacts between distinct 250kb bins.
        np.savez_compressed(out / f'{chrom}_counts.npz', counts=mats)
        sym = mats + mats.transpose(0,2,1)
        coverage.append(sym.sum(axis=2))
        u,v = np.triu_indices(nb,1)
        contacts.append(mats[:,u,v])
        insulation=[]
        for window in (2,5,10):
            x=np.zeros((n,nb),np.float32)
            for b in range(1,nb):
                x[:,b]=mats[:,max(0,b-window):b,b:min(nb,b+window)].sum(axis=(1,2))
            insulation.append(x)
        local.append(np.concatenate(insulation,axis=1))
        print(f'{chrom}: {n} cells, {nb} bins',flush=True)
    np.savez_compressed(out/'features.npz', decay=decay, edges=edges,
                        coverage=np.concatenate(coverage,axis=1),
                        insulation=np.concatenate(local,axis=1),
                        contacts=np.concatenate(contacts,axis=1))
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Finished label-free extraction',flush=True)

if __name__=='__main__':
    main()
