for i in $(seq 1 40); do
  OnTAD/OnTAD/src/OnTAD single_cells/h1esc/cell${i}.txt -o single_cells/h1esc/tads/cell${i} -penalty 0.2
done