import pandas as pd
import ast
from zss import Node, simple_distance

def build_tree(edges):
    if not edges: return None
    nodes = {}
    items = set()
    for p, c in edges:
        items.add(p)
        items.add(c)
    for item in items:
        nodes[item] = Node(str(item))
    for p, c in edges:
        nodes[p].addkid(nodes[c])
    all_children = {c for p, c in edges}
    roots = [nodes[n] for n in items if n not in all_children]
    return roots[0] if roots else None

ref_edges = [(1, 2), (2, 4), (4, 8), (8, 64)]
ref_tree = build_tree(ref_edges)

csv_path = 'results/scptr_tree_distance/all_tree_distances.csv'
df = pd.read_csv(csv_path)
processed = 0
changed = 0

for i, row in df.iterrows():
    try:
        edges = ast.literal_eval(row['spanning_tree_edges'])
        tree = build_tree(edges)
        if tree:
            dist = float(simple_distance(ref_tree, tree))
            if abs(dist - row['tree_distance']) > 1e-9:
                df.at[i, 'tree_distance'] = dist
                changed += 1
        processed += 1
    except Exception:
        continue

df.to_csv(csv_path, index=False)
print(f"Processed rows: {processed}")
print(f"Rows changed: {changed}")
target = df[(df['chrom'] == 'chr5') & (df['tol'] == 1) & (df['resolution'] == 1000)]
val = target['tree_distance'].values[0] if not target.empty else None
print(f"chr5 tol=1 1000kb tree_distance: {val}")
