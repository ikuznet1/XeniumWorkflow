# Methods


## Data

1 sample(s) were loaded (S0: sectionA, 1,200 cells), 1,200 cells and 300 genes in the primary sample's panel.

## Banksy-style spatial domains

Banksy-style spatial domains on the Xenium segmentation: a native implementation that follows the construction of the published BANKSY method - each cell's own expression, the Gaussian-weighted mean expression of its k nearest cells in space and the azimuthal Gabor filter of its 2k nearest cells, each block z-scaled per gene and mixed by lambda - followed by PCA, a nearest-neighbour graph in PC space and Leiden clustering (KMeans when Leiden is not installed; the record's status line says which). It is not the Banksy package and has not been validated against the reference implementation. The clusters are an exploratory partition, not a test. [1] [2]

Original counts were used.

Parameters: lam = 0.8; k = 15; resolution = 1; seed = 0; resolved.n_pcs = 20.

Random seeds: seed = 0.

As reported by the app:

> Banksy-style λ=0.8: 7 clusters (from the cache).

Software: Python 3.11.6; numpy 1.26.0; scipy 1.11.0; pandas 2.1.0; scikit-learn 1.3.0. Record prov-20260102-030405-banksy (loaded from the cache, 1.5 s).

## References

1. Singhal V, Chou N, Lee J, et al. BANKSY unifies cell typing and tissue domain segmentation for scalable spatial omics data analysis. Nat Genet 56, 431-441 (2024).
2. Traag VA, Waltman L, van Eck NJ. From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 9, 5233 (2019).

Entries marked [verify] could not be confirmed when the app's reference list was written: check them against the publication before citing.

## All records

| When | Analysis | Segmentation | Counts | Source | Seconds | Record |
|---|---|---|---|---|---|---|
| 2026-01-02 03:04:05 | Banksy-style spatial domains | xenium | original | cache | 1.5 | prov-20260102-030405-banksy |
