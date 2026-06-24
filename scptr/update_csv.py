import ast
import csv
from pathlib import Path

from zss import Node, simple_distance


def build_reference_chain_tree():
    root = Node("1")
    n2 = Node("2")
    n4 = Node("4")
    n8 = Node("8")
    n64 = Node("64")
    root.addkid(n2)
    n2.addkid(n4)
    n4.addkid(n8)
    n8.addkid(n64)
    return root


def build_rooted_tree_from_edge_labels(edge_labels):
    adjacency = {}

    for edge in edge_labels:
        if not isinstance(edge, str) or "-" not in edge:
            continue
        u, v = edge.split("-", 1)
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)

    if not adjacency:
        return None

    root = "1" if "1" in adjacency else sorted(adjacency.keys(), key=str)[0]

    def build(node, parent=None):
        t = Node(str(node))
        for nbr in sorted(adjacency.get(node, []), key=str):
            if nbr == parent:
                continue
            t.addkid(build(nbr, node))
        return t

    return build(root)


def resolve_csv_path():
    candidates = [
        Path("results/scptr_tree_distance/all_tree_distances.csv"),
        Path("results/scptr_tree_distance/all_tree_distance.csv"),
        Path("scptr/results/scptr_tree_distance/all_tree_distances.csv"),
        Path("scptr/results/scptr_tree_distance/all_tree_distance.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find tree-distance CSV. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def main():
    csv_path = resolve_csv_path()
    ref_tree = build_reference_chain_tree()

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    processed = 0
    changed = 0
    target_dist = None

    for row in rows:
        processed += 1
        old_val = row.get("tree_distance", "")
        edges_raw = row.get("spanning_tree_edges", "")

        try:
            edge_labels = ast.literal_eval(edges_raw)
        except Exception:
            continue

        if not isinstance(edge_labels, list):
            continue

        inferred_tree = build_rooted_tree_from_edge_labels(edge_labels)
        if inferred_tree is None:
            continue

        new_dist = float(
            simple_distance(
                inferred_tree,
                ref_tree,
                get_children=lambda n: n.children,
                get_label=lambda n: n.label,
            )
        )

        row["tree_distance"] = str(new_dist)
        if str(old_val) != row["tree_distance"]:
            changed += 1

        if (
            row.get("chromosome") == "chr5"
            and row.get("tolerance") == "1"
            and row.get("resolution") == "1000kb"
        ):
            target_dist = row["tree_distance"]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV: {csv_path}")
    print(f"Processed rows: {processed}")
    print(f"Rows changed: {changed}")
    print(f"chr5 tol=1 1000kb tree_distance: {target_dist}")


if __name__ == "__main__":
    main()
