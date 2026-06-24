import pandas as pd
from pathlib import Path
import numpy as np

base = Path('.')
res5 = 5000
res250 = 250000


def process_loops(prefix: str, infile: Path, tag: str):
    raw = pd.read_csv(infile, sep='\t', header=None)
    if raw.shape[1] >= 10:
        raw = raw.iloc[:, :10]
        raw.columns = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'score', 'IF', 'P2LL', 'label']
    else:
        raw = raw.iloc[:, :9]
        raw.columns = ['chr1', 'start1', 'end1', 'chr2', 'start2', 'end2', 'score', 'IF', 'P2LL']
    df = raw.copy()

    for c in ['start1', 'end1', 'start2', 'end2', 'score', 'IF', 'P2LL']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['start1', 'end1', 'start2', 'end2', 'score', 'IF', 'P2LL']).copy()

    df['bin1_5kb'] = (df['start1'] // res5).astype(int)
    df['bin2_5kb'] = (df['start2'] // res5).astype(int)
    df['bin1_250kb'] = (df['start1'] // res250).astype(int)
    df['bin2_250kb'] = (df['start2'] // res250).astype(int)

    out_bins = base / f'{prefix}.{tag}_with_bins.tsv'
    cols = [
        'chr1',
        'start1',
        'end1',
        'bin1_5kb',
        'bin1_250kb',
        'chr2',
        'start2',
        'end2',
        'bin2_5kb',
        'bin2_250kb',
        'score',
        'IF',
        'P2LL',
    ]
    if 'label' in df.columns:
        cols.append('label')
    df[cols].to_csv(out_bins, sep='\t', index=False)

    g = (
        df.groupby(['chr1', 'bin1_250kb', 'chr2', 'bin2_250kb'], as_index=False)
        .agg(score=('score', 'max'), IF=('IF', 'mean'), P2LL=('P2LL', 'mean'))
    )

    g['start1'] = g['bin1_250kb'] * res250
    g['end1'] = g['start1'] + res250
    g['start2'] = g['bin2_250kb'] * res250
    g['end2'] = g['start2'] + res250

    g = g[['chr1', 'start1', 'end1', 'bin1_250kb', 'chr2', 'start2', 'end2', 'bin2_250kb', 'score', 'IF', 'P2LL']]
    out250 = base / f'{prefix}.{tag}_250kb.tsv'
    g.to_csv(out250, sep='\t', index=False)

def transform_one(prefix: str):
    # ---------- BoundaryScore ----------
    boundary_path = base / f'{prefix}_BoundaryScore.bed'
    b = pd.read_csv(
        boundary_path,
        sep='\t',
        header=None,
        names=['chr', 'start', 'end', 'lScore', 'lBoundary', 'rScore', 'rBoundary'],
    )

    b['bin5kb'] = (b['start'] // res5).astype(int)
    b['bin250kb'] = (b['start'] // res250).astype(int)

    b5_out = base / f'{prefix}_BoundaryScore_with_bins.tsv'
    b[
        ['chr', 'start', 'end', 'bin5kb', 'bin250kb', 'lScore', 'lBoundary', 'rScore', 'rBoundary']
    ].to_csv(b5_out, sep='\t', index=False)

    b250 = (
        b.groupby(['chr', 'bin250kb'], as_index=False)
        .agg(
            lScore=('lScore', 'mean'),
            lBoundary=('lBoundary', 'max'),
            rScore=('rScore', 'mean'),
            rBoundary=('rBoundary', 'max'),
        )
    )
    b250['start'] = b250['bin250kb'] * res250
    b250['end'] = b250['start'] + res250
    b250 = b250[['chr', 'start', 'end', 'bin250kb', 'lScore', 'lBoundary', 'rScore', 'rBoundary']]

    b250_out = base / f'{prefix}_BoundaryScore_250kb.tsv'
    b250.to_csv(b250_out, sep='\t', index=False)

    # ---------- TAD ----------
    tad_path = base / f'{prefix}_TAD.bed'
    t = pd.read_csv(tad_path, sep='\t', header=None, names=['chr', 'start', 'end'])

    if len(t) > 0:
        t['start_bin5kb'] = (t['start'] // res5).astype(int)
        t['end_bin5kb'] = ((t['end'] - 1) // res5).astype(int)
        t['start_bin250kb'] = (t['start'] // res250).astype(int)
        t['end_bin250kb'] = ((t['end'] - 1) // res250).astype(int)

        t5_out = base / f'{prefix}_TAD_with_bins.tsv'
        t[
            ['chr', 'start', 'end', 'start_bin5kb', 'end_bin5kb', 'start_bin250kb', 'end_bin250kb']
        ].to_csv(t5_out, sep='\t', index=False)

        # Snap to 250kb grid
        start250 = (t['start'] // res250) * res250
        end250 = ((t['end'] + res250 - 1) // res250) * res250
        end250 = np.where(end250 <= start250, start250 + res250, end250)

        t250 = pd.DataFrame({'chr': t['chr'], 'start': start250.astype(int), 'end': end250.astype(int)})
        t250 = t250.drop_duplicates().sort_values(['chr', 'start', 'end']).reset_index(drop=True)
        t250['start_bin250kb'] = (t250['start'] // res250).astype(int)
        t250['end_bin250kb'] = ((t250['end'] - 1) // res250).astype(int)

        t250_out = base / f'{prefix}_TAD_250kb.tsv'
        t250.to_csv(t250_out, sep='\t', index=False)

    # ---------- Loop files ----------
    process_loops(prefix, base / f'{prefix}.loop_candidates.bedpe', 'loop_candidates')
    process_loops(prefix, base / f'{prefix}.loops.bedpe', 'loops')


for prefix in ['MPP/chr4/MPP_refhic_chr4', 'CMP/chr4/CMP_refhic_chr4']:
    transform_one(prefix)

print('Done')
