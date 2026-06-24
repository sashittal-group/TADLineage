from pathlib import Path
import subprocess

pred_root = Path('/Users/PSLabMonisha/Desktop/tads/refhic_predictions')
conv_dir = pred_root / 'converted_mcool'


def refhic_cmd(*args):
    return ['conda', 'run', '-n', 'refhic', 'refhic', *args]


def find_sample_mcool(sample: str) -> Path:
    candidates = [
        conv_dir / f'GSE152918_{sample}.mcool',
        Path(f'/Users/PSLabMonisha/Desktop/tads/refhic_predictions/converted_mcool/GSE152918_{sample}.mcool'),
        Path(f'/Users/PSLabMonisha/Desktop/tads/hemato_cells/GSE152918_{sample}.mcool'),
    ]
    hit = next((p for p in candidates if p.exists()), None)
    if hit is None:
        raise FileNotFoundError(
            f'{sample} mcool not found. Checked:\n' + '\n'.join(str(p) for p in candidates)
        )
    return hit


def run_one(sample: str, chrom: str) -> None:
    target_mcool = find_sample_mcool(sample)
    output_dir = pred_root / sample / chrom
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_dir / f'{sample}_refhic_{chrom}'
    loop_candidates = output_dir / f'{sample}_refhic_{chrom}.loop_candidates.bedpe'
    loops = output_dir / f'{sample}_refhic_{chrom}.loops.bedpe'

    print(f'Running RefHiC TAD/boundary prediction for {sample} on {chrom}...')
    subprocess.run(
        refhic_cmd(
            'tad',
            'pred',
            '--cpu',
            'True',
            '--chrom',
            chrom,
            '-t',
            '1',
            str(target_mcool),
            str(prefix),
        ),
        check=True,
    )

    print(f'Running RefHiC loop candidate prediction for {sample} on {chrom}...')
    subprocess.run(
        refhic_cmd(
            'loop',
            'pred',
            '--cpu',
            'True',
            '--chrom',
            chrom,
            '-t',
            '1',
            str(target_mcool),
            str(loop_candidates),
        ),
        check=True,
    )

    print(f'Pooling final loops for {sample} on {chrom}...')
    subprocess.run(
        refhic_cmd(
            'loop',
            'pool',
            str(loop_candidates),
            str(loops),
        ),
        check=True,
    )

    print('Generated:')
    print(prefix.with_name(prefix.name + '_BoundaryScore.bed'))
    print(prefix.with_name(prefix.name + '_TAD.bed'))
    print(loop_candidates)
    print(loops)


jobs = [
    ('MPP', 'chr4'),
    ('CMP', 'chr4'),
]

for sample_name, chrom_name in jobs:
    run_one(sample_name, chrom_name)

print('All requested jobs completed.')
