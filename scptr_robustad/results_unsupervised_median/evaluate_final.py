"""Read stage metadata only after the standalone fit has completed."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from evaluate import truth

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='final')
    args=parser.parse_args()
    folder=(HERE/args.output).resolve()
    if not folder.is_relative_to(HERE): raise ValueError('Output must stay inside the experiment')
    pred=np.load(folder/'fit.npz')['labels']
    labels=truth()
    result={'n_cells':len(labels),'ARI':float(adjusted_rand_score(labels,pred)),
            'NMI':float(normalized_mutual_info_score(labels,pred)),
            'k':len(set(pred)),'stage_labels_used_in_fit':False,
            'status':'Exploratory benchmark; hyperparameters were investigated using ARI.'}
    (folder/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
