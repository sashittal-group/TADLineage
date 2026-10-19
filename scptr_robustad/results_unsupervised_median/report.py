"""Evaluation and figures, intentionally separate from every fitting module."""
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import truth

HERE=Path(__file__).resolve().parent
STAGES=['1-cell','2-cell','4-cell','8-cell','64-cell']

def main():
    out=HERE/'final'
    y=truth()
    ids=json.loads((HERE/'data/cells.json').read_text())
    fit=np.load(out/'fit.npz'); pred=fit['labels']
    metric=lambda z:float(adjusted_rand_score(y,z))
    rows=[]; predictions={}
    for path in sorted((HERE/'hierarchies').glob('*/run.json')):
        r=json.loads(path.read_text()); z=np.load(path.parent/'labels.npy')
        r['ARI']=metric(z); r['NMI']=float(normalized_mutual_info_score(y,z))
        rows.append(r); predictions[r['run']]=z
    hdir=HERE/'hierarchies'
    (hdir/'evaluation.json').write_text(json.dumps(rows,indent=2))
    (hdir/'runs.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('ARI','NMI')} for r in rows],indent=2))
    np.savez_compressed(hdir/'predictions.npz',**predictions)
    # Controls are not candidates for model selection.
    candidates=[r for r in rows if not r['run'].startswith(('control_','validation_'))]
    selected=max(candidates,key=lambda r:r['silhouette'])
    (hdir/'selected.json').write_text(json.dumps({k:v for k,v in selected.items() if k not in ('ARI','NMI')},indent=2))
    with (HERE/'experiment_summary.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t'); writer.writerow(['run','ARI','NMI','RF_silhouette','objective'])
        for r in rows: writer.writerow([r[k] for k in ('run','ARI','NMI','silhouette','objective')])
    confusion=np.array([[np.sum((y==s)&(pred==c)) for c in range(5)] for s in STAGES])
    ri,ci=linear_sum_assignment(-confusion)
    mapping={int(c):STAGES[int(r)] for r,c in zip(ri,ci)}
    with (out/'evaluation_assignments.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t'); writer.writerow(['cell_id','stage_for_evaluation','cluster'])
        writer.writerows(zip(ids,y,pred))
    pair_a,pair_b=np.triu_indices(len(y),1)
    same=y[pair_a]==y[pair_b]
    distance_summary={}
    for name,path in [('raw',hdir/'raw/trees.npz'),('imputed',out/'trees.npz')]:
        t=sparse.load_npz(path); size=np.asarray(t.sum(axis=1)).ravel()
        denom=size[:,None]+size[None,:]
        d=(denom-2*(t@t.T).toarray())/np.maximum(denom,1)
        values=d[pair_a,pair_b]
        distance_summary[name]={'within_stage':float(np.mean(values[same])),
                                'between_stage':float(np.mean(values[~same]))}
    w=fit['weights'].copy(); np.fill_diagonal(w,0)
    purity=float(np.mean((w*(y[:,None]==y[None,:])).sum(axis=1)/w.sum(axis=1)))
    restart_ari=[metric(z) for z in fit['restart_labels']]
    result={'ARI':metric(pred),'NMI':float(normalized_mutual_info_score(y,pred)),
            'n_cells':len(y),'stage_counts':dict(Counter(y)),
            'confusion_rows':STAGES,'confusion_columns':list(range(5)),
            'confusion':confusion.tolist(),'cluster_stage_mapping_for_display_only':mapping,
            'neighbor_same_stage_weight_for_evaluation_only':purity,
            'normalized_RF':distance_summary,'restart_ARI':restart_ari,
            'restart_best_objective_selected_without_labels':True,
            'exploratory_hierarchy_silhouette_selection':{'run':selected['run'],'ARI':selected['ARI']}}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    ax=axes[0,0]
    ax.imshow(confusion,cmap='Blues')
    for i in range(5):
        for j in range(5): ax.text(j,i,str(confusion[i,j]),ha='center',va='center',color='white' if confusion[i,j]>90 else 'black')
    ax.set(xticks=range(5),yticks=range(5),yticklabels=STAGES,xlabel='Median cluster',title=f'All 508 real cells · ARI {result["ARI"]:.3f}')
    metrics={r['run']:r['ARI'] for r in rows}
    names=['No imputation','Global pooling','Shuffled graph','Spectral only','Graph + median']
    spectral=np.load(HERE/'signals_refined/predictions.npz')['contacts__d10__spectral10']
    vals=[metrics['raw'],metrics['global_pool'],metrics['control_shuffled_neighbors'],metric(spectral),result['ARI']]
    ax=axes[0,1]; ax.barh(names,vals,color=['#b0b7c3']*3+['#d99736','#3676ab']); ax.set(xlim=(-.05,1.05),xlabel='ARI',title='Ablations (k = 5)')
    for i,v in enumerate(vals): ax.text(v+.01,i,f'{v:.3f}',va='center')
    grid=np.zeros((3,3))
    for i,nn in enumerate((8,10,12)):
        for j,t in enumerate((3,5,8)):
            grid[i,j]=metrics[f'contacts_d10_nn{nn}_self0.05_steps{t}_sigma0.7_oe0_symmetric_spectral']
    ax=axes[1,0]; ax.imshow(grid,vmin=0,vmax=1,cmap='YlGnBu')
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{grid[i,j]:.3f}',ha='center',va='center',color='white' if grid[i,j]>.7 else 'black')
    ax.set(xticks=range(3),xticklabels=[3,5,8],yticks=range(3),yticklabels=[8,10,12],xlabel='Diffusion steps',ylabel='Nearest neighbors',title='Parameter sensitivity; all runs shown')
    ax=axes[1,1]
    pos=np.arange(2)
    ax.bar(pos-.18,[distance_summary[s]['within_stage'] for s in ('raw','imputed')],.36,label='Within stage')
    ax.bar(pos+.18,[distance_summary[s]['between_stage'] for s in ('raw','imputed')],.36,label='Between stages')
    ax.set(xticks=pos,xticklabels=['Raw hierarchies','Imputed hierarchies'],ylabel='Mean normalized RF distance',ylim=(0,1),title='Do same-stage hierarchies become closer?'); ax.legend()
    fig.savefig(out/'benchmark.png',dpi=180); fig.savefig(out/'benchmark.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    coords=fit['embedding'][:,:2]
    colors=plt.get_cmap('tab10').colors
    for j,s in enumerate(STAGES):
        take=y==s
        axes[0].scatter(*coords[take].T,s=15,color=colors[j],label=s,alpha=.8)
        take=np.array([mapping[int(c)]==s for c in pred])
        axes[1].scatter(*coords[take].T,s=15,color=colors[j],label=s,alpha=.8)
    for ax in axes: ax.set(xlabel='Contact PC1',ylabel='Contact PC2')
    axes[0].set_title('Stage labels (evaluation only)'); axes[0].legend(fontsize=8)
    axes[1].set_title('Median clusters (colors matched after fitting)')
    fig.savefig(out/'embedding.png',dpi=180); plt.close(fig)
    summary={**result,'ablations':dict(zip(names,vals)),
             'sensitivity_ARI':grid.tolist(),
             'cross_chromosome_ARI':{r['run']:r['ARI'] for r in rows if r['run'].startswith('validation_')}}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
