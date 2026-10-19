"""Select neighbor count/diffusion length using held-out contacts, without labels.

Three fixed 50/50 binomial splits. Refit PCA and graph on training counts.
Score equal-cell, equal-chromosome held-out multinomial cross entropy.
Use a fixed 1% global training prior for unseen entries. Lower is better.
"""
import json
import numpy as np
from signals import pca,normalize
from hierarchy import diffusion_weights
from extract import HERE,CHROMS

def main():
    folder=HERE/'reconstruction'; folder.mkdir(exist_ok=True)
    configs=[(nn,t) for nn in (8,10,12) for t in (3,5,8)]
    rows=[]
    for seed in (0,1,2):
        rng=np.random.default_rng(seed)
        train=[]; test=[]
        for chrom in CHROMS:
            m=np.load(HERE/f'data/{chrom}_counts.npz')['counts']
            a,b=np.triu_indices(m.shape[1],1)
            counts=m[:,a,b].astype(int)
            tr=rng.binomial(counts,.5).astype(np.float32)
            train.append(tr); test.append(counts.astype(np.float32)-tr)
        x=pca(np.sqrt(normalize(np.concatenate(train,axis=1))),10)
        for nn,t in configs:
            w=diffusion_weights(x,nn,.05,t)
            score=[]
            for tr,te in zip(train,test):
                prior=tr.sum(axis=0)+1e-3
                prior/=prior.sum()
                pred=.99*(w@normalize(tr))+.01*prior[None,:]
                ce=-(te*np.log(np.maximum(pred,1e-15))).sum(axis=1)/np.maximum(te.sum(axis=1),1)
                score.append(float(np.mean(ce)))
            row={'seed':seed,'neighbors':nn,'steps':t,'cross_entropy':float(np.mean(score))}
            rows.append(row)
            print(row,flush=True)
    summary=[]
    for nn,t in configs:
        values=[r['cross_entropy'] for r in rows if r['neighbors']==nn and r['steps']==t]
        summary.append({'neighbors':nn,'steps':t,'mean_cross_entropy':float(np.mean(values)),
                        'std_cross_entropy':float(np.std(values))})
    (folder/'splits.json').write_text(json.dumps(rows,indent=2))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    (folder/'selected.json').write_text(json.dumps(min(summary,key=lambda r:r['mean_cross_entropy']),indent=2))

if __name__=='__main__': main()
