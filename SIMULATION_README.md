# TAD Simulation: Random Tree & Interval Generation

## Overview
This modified version of `check_lp_constrained.py` runs a simulation that generates random tree structures and random interval assignments to test for cases where the LP relaxation solution differs from the brute-force solution and is non-integral.

## Key Features

### 1. Random Tree Generation
- **Function**: `generate_random_tree(num_internal, num_leaves)`
- Generates a tree with specified number of internal nodes and leaves
- Node 0 is always the root
- Internal nodes are numbered 0 to `num_internal-1`
- Leaf nodes are numbered `num_internal` to `num_internal + num_leaves - 1`
- Each node is randomly connected to create a valid tree structure

### 2. Random Interval Generation
- **Function**: `generate_random_intervals(min_val, max_val, num_intervals, min_length=1)`
- Generates random intervals within the range [min_val, max_val]
- Creates diverse intervals of varying lengths
- Returns interval strings in format "[start,end]"

### 3. Laminar Interval Assignment
- **Function**: `generate_random_laminar_intervals_from_list(leaf_nodes, available_intervals, max_per_leaf=5)`
- Randomly assigns intervals to each leaf node
- Ensures laminarity constraint is satisfied (no crossing intervals at any leaf)
- Falls back to single interval if laminar set cannot be found
- Respects `max_per_leaf` parameter to control interval density per leaf

### 4. Simulation Loop with Varying Tree Sizes
- **Function**: `run_simulation(num_simulations, min_internal, max_internal, min_leaves, max_leaves, min_val, max_val, num_intervals, max_per_leaf, Ct, Cb)`
- Runs multiple simulations with different random trees and intervals
- Each simulation randomly selects:
  - Number of internal nodes from `[min_internal, max_internal]`
  - Number of leaf nodes from `[min_leaves, max_leaves]`
- For each simulation:
  - Generates a unique random tree structure
  - Solves the LP relaxation
  - Computes the optimal solution via brute-force Sankoff algorithm
  - Checks for integrality of LP solution
  - **Terminates immediately** if it finds a case where:
    - LP cost ≠ Brute-force cost (difference > 1e-4)
    - AND LP solution is not integral

## Configuration Parameters

Edit these in the `if __name__ == "__main__":` section:

```python
NUM_SIMULATIONS = 1000        # Number of simulations to run
MIN_INTERNAL = 3              # Minimum number of internal nodes (including root)
MAX_INTERNAL = 100            # Maximum number of internal nodes
MIN_LEAVES = 5                # Minimum number of leaf nodes
MAX_LEAVES = 100              # Maximum number of leaf nodes
MIN_VAL = 1                   # Minimum value for intervals
MAX_VAL = 50                  # Maximum value for intervals
NUM_INTERVALS = 15            # Number of different intervals to generate
MAX_PER_LEAF = 10             # Maximum number of intervals per leaf
Ct = 2.0                      # Cost of TAD changes
Cb = 1.5                      # Cost of boundary changes
```

## Usage

```bash
python check_lp_constrained.py
```

## Output

### Normal Case (No Mismatch Found)
```
Running 1000 simulations...
Tree structure: 3-100 internal nodes, 5-100 leaves
Intervals: range [1, 50], 15 total intervals
Max intervals per leaf: 10
Costs: Ct=2.0, Cb=1.5

Simulation 1 (Internal=4, Leaves=7)... OK (LP=4.00, BF=4.00, Integral=True)
Simulation 2 (Internal=5, Leaves=9)... OK (LP=6.00, BF=6.00, Integral=True)
Simulation 3 (Internal=3, Leaves=12)... OK (LP=5.50, BF=5.50, Integral=True)
...
Completed 1000 simulations. No mismatches found.
```

### Mismatch Found
```
Simulation 42 (Internal=6, Leaves=8)... 

*** FOUND MISMATCH! ***
Simulation #42
LP cost: 3.5000
Brute-force cost: 4.0000
Is integral: False

Tree edges: [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (0, 7)]

Leaf intervals:
  Node 3: ['[1,3]', '[4,6]']
  Node 4: ['[1,6]']
  Node 5: ['[2,5]', '[7,9]']
  Node 6: ['[1,2]', '[8,10]']
  Node 7: ['[3,5]']

LP solution (non-integral values at internal nodes):
  Node 0:
    x[0,(1,6)] = 0.5000
    x[0,(2,5)] = 0.5000
  Node 1:
    x[1,(1,6)] = 0.3333
    x[1,(2,5)] = 0.6667
  Node 2:
    x[2,(1,6)] = 0.5000
    x[2,(4,6)] = 0.5000
```

## How It Works

1. **Tree Generation**: Each simulation creates a new random tree structure ensuring:
   - Valid parent-child relationships
   - All nodes are connected (forming a tree)
   - Proper separation between internal and leaf nodes
   - Variety in tree topology across simulations

2. **Interval Generation**: Random intervals are created ensuring:
   - Diversity in interval sizes
   - Proper bounds within [MIN_VAL, MAX_VAL]
   - No duplicate intervals

3. **Laminar Assignment**: At each leaf:
   - Randomly selects a subset of available intervals (up to `MAX_PER_LEAF` intervals)
   - Verifies laminarity (no crossing intervals)
   - Retries if needed to find valid laminar sets
   - Falls back to single interval if no laminar set is found

4. **LP vs Brute-Force Comparison**:
   - LP solver finds relaxed solution (variables can be fractional)
   - Sankoff algorithm finds exact optimal integer solution
   - If they differ AND LP is non-integral, the simulation stops and displays the instance

## Customization

To adjust randomness and tree structure:
- Remove or change `random.seed(42)` for different random sequences
- Adjust `MIN_INTERNAL` and `MAX_INTERNAL` to control the range of internal nodes per simulation
- Adjust `MIN_LEAVES` and `MAX_LEAVES` to control the range of leaf nodes per simulation
- Change `MAX_PER_LEAF` to control the maximum number of intervals assigned to each leaf
- Change `MIN_VAL`, `MAX_VAL`, and `NUM_INTERVALS` for different interval ranges
- Modify `Ct` and `Cb` to change the cost weights for TAD changes and boundary changes

## Notes

- The simulation terminates as soon as a problematic instance is found
- Brute-force algorithm may be slow for large numbers of intervals (exponential in the number of distinct intervals)
- Gurobi output is suppressed (`OutputFlag=0`) for cleaner simulation output
- Tree sizes vary randomly for each simulation, making the test coverage more comprehensive
- Larger tree sizes increase computational complexity; consider adjusting `MAX_INTERNAL` and `MAX_LEAVES` accordingly
- Each simulation is independent; setting `random.seed()` helps with reproducibility

## Performance Considerations

- **Large tree sizes**: Setting `MAX_INTERNAL` and `MAX_LEAVES` to large values increases computation time significantly
- **Many intervals**: Increasing `NUM_INTERVALS` increases the state space for the brute-force algorithm (exponential growth)
- **Max per leaf**: Higher `MAX_PER_LEAF` values result in denser interval assignments and potentially longer solving times
- For quick testing, use smaller ranges: `MIN_INTERNAL=2, MAX_INTERNAL=5, MIN_LEAVES=3, MAX_LEAVES=8`
