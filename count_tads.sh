stages=("1-cell" "2-cell" "4-cell" "8-cell" "64-cell")
base_dir="scptr_robustad/cell_differentiation_robustad_ontad"

echo -e "Stage\tRes\tTotal_GSM\tWith_TAD\tMissing\tEmpty"

for stage in "${stages[@]}"; do
    stage_path="$base_dir/$stage"
    if [ ! -d "$stage_path" ]; then continue; fi
    
    gsms=($(find "$stage_path" -maxdepth 1 -mindepth 1 -type d))
    total_gsm=${#gsms[@]}
    
    for res in "500kb" "1000kb"; do
        with_tad=0
        missing=0
        empty=0
        
        for gsm_path in "${gsms[@]}"; do
            tad_file="$gsm_path/chr1/$res/tads.tad"
            if [ -f "$tad_file" ]; then
                with_tad=$((with_tad + 1))
                if [ ! -s "$tad_file" ]; then
                    empty=$((empty + 1))
                fi
            else
                missing=$((missing + 1))
            fi
        done
        
        echo -e "$stage\t$res\t$total_gsm\t$with_tad\t$missing\t$empty"
    done
done
