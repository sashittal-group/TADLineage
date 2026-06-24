# Clustering Approaches: Embeddings vs. Compartments

This directory now includes two clustering pipelines for scHi-C data analysis:

## 1. Embedding-Based Clustering (`cluster_embeddings.py`)

**Source of information:** Neural network embeddings from the imputation training phase

**Process:**
- Uses learned cell embeddings (low-dimensional representations)
- Applies KMeans clustering on embedding space
- Fast and directly uses learned features

**Output:**
- `cluster_results.txt` - Clustering metrics (ARI, NMI, Silhouette) and cluster assignments

**Run:**
```bash
python cluster_embeddings.py
```

---

## 2. Compartment-Based Clustering (`cluster_compartments.py`) ⭐ NEW

**Source of information:** A/B compartment profiles from imputed contact maps

**Process:**
1. Loads imputed contact maps (from `temp/*.hdf5` files)
2. Calculates A/B compartment score per cell for each chromosome
   - Uses the Lieberman-Aiden et al. method (PCA-based)
   - First principal component captures compartment signal
3. Concatenates compartment profiles across all chromosomes
4. Performs KMeans clustering on concatenated compartment features
5. Visualizes results with UMAP

**Why this approach?**
- Uses genomic information directly from imputed maps
- Captures higher-order 3D structure (A/B compartments)
- Complements embedding-based approach by using imputed data

**Output:**
- `compartment_clustering_results.txt` - Metrics and cluster assignments
- `compartment_clustering_figure.png` - Visualization with UMAP

**Run:**
```bash
python cluster_compartments.py
```

---

## 3. Method Comparison (`compare_clustering_methods.py`) ⭐ NEW

**Compares both approaches side-by-side**

**What it does:**
1. Loads embeddings from neural network
2. Loads compartment features from imputed maps
3. Performs clustering with both methods
4. Generates comprehensive comparison figure with 4 panels:
   - Embedding-based clustering result
   - True cell types in embedding space
   - Compartment-based clustering result
   - True cell types in compartment space

**Output:**
- `clustering_comparison_figure.png` - 4-panel comparison figure
- Console output with detailed metrics for both methods

**Run:**
```bash
python compare_clustering_methods.py
```

---

## Workflow Summary

```
Raw scHi-C Data
    ↓
    ├─→ [Embedding Path]
    │   ├─ Neural Network Training
    │   ├─ Generate Embeddings
    │   └─ cluster_embeddings.py → embedding-based clusters
    │
    └─→ [Compartment Path]
        ├─ Imputation Training
        ├─ Generate Imputed Maps
        ├─ Extract A/B Compartments (from imputed maps)
        └─ cluster_compartments.py → compartment-based clusters
```

---

## Key Differences

| Feature | Embeddings | Compartments |
|---------|-----------|-------------|
| **Source** | Learned representations | Imputed contact maps |
| **Feature type** | Low-dimensional latent space | Biological (A/B compartments) |
| **Dimension** | 64-256 dims | ~500+ (size × #chroms) |
| **Interpretability** | Abstract patterns | Direct genomic features |
| **Computational cost** | Fast | Moderate (requires imputation) |
| **Use case** | General clustering | Structure-aware clustering |

---

## Example Results Interpretation

### Metrics Explained:
- **ARI (Adjusted Rand Index)**: -1 (opposite) to 1 (perfect match) with true labels
- **NMI (Normalized Mutual Info)**: 0 (independent) to 1 (perfect match) with true labels  
- **Silhouette Score**: -1 to 1, higher is better (compact, well-separated clusters)

### Visualization:
- **Top panels**: UMAP visualization in embedding/compartment space, colored by predicted clusters and true cell types
- **Perfect clustering**: Predicted cluster colors align with true cell types
- **Partial match**: Some cluster mixing indicates moderate separation

---

## Configuration

Ensure `config.json` includes:
- `temp_dir`: Path to imputed data files
- `embedding_name`: Name of embedding files (default: "embed")
- `chrom_list`: List of chromosomes to analyze

Example:
```json
{
  "temp_dir": "./temp",
  "embedding_name": "embed",
  "chrom_list": ["chr1", "chr2", "chr3", ...]
}
```

---

## Dependencies

```
numpy, pandas, scipy
scikit-learn (clustering, metrics, preprocessing)
h5py (reading imputed data)
matplotlib, seaborn, umap-learn (visualization)
```

Install with:
```bash
pip install numpy pandas scipy scikit-learn h5py matplotlib seaborn umap-learn
```

---

## Future Extensions

- [ ] Use TAD information alongside compartments
- [ ] Multi-scale compartment hierarchy clustering
- [ ] Integration with cell-type-specific compartment patterns
- [ ] Comparison with other compartment calling methods

