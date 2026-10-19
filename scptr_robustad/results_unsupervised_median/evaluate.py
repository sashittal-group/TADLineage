"""Post-hoc evaluation only. Do not import this module from any fitting script."""
from pathlib import Path
import csv
import json
import argparse
import numpy as np
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

HERE=Path(__file__).resolve().parent

def truth():
    ids=json.loads((HERE/'data/cells.json').read_text())
    with (HERE.parents[1]/'scptr/GSE129029_metadata.csv').open() as f:
        labels={r['GSM']:r['CELL_STAGE'] for r in csv.DictReader(f)}
    return np.array([labels[s] for s in ids])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',nargs='?',default='signals')
    args=parser.parse_args()
    out=HERE/args.folder
    pred=np.load(out/'predictions.npz')
    labels=truth()
    runs=json.loads((out/'runs.json').read_text())
    for r in runs:
        y=pred[r['run']]
        r['ARI']=float(adjusted_rand_score(labels,y))
        r['NMI']=float(normalized_mutual_info_score(labels,y))
    (out/'evaluation.json').write_text(json.dumps(runs,indent=2))
    for r in sorted(runs,key=lambda r:r['ARI'],reverse=True)[:15]:
        print(f"{r['run']}: ARI={r['ARI']:.4f}, silhouette={r.get('silhouette',0):.4f}")
    selected=json.loads((out/'selected.json').read_text())
    print('Unsupervised selection:',next(r for r in runs if r['run']==selected['run']))

if __name__=='__main__': main()
