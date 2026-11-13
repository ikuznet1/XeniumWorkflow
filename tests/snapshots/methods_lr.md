# Methods


## Data

1 sample(s) were loaded (S0: sectionA, 1,200 cells), 1,200 cells and 300 genes in the primary sample's panel.

## Ligand-receptor scores

Ligand-receptor scores between cell groups on the Xenium segmentation: the score of a pair for sender type S and receiver type R is the mean of the mean log-normalised expression of the ligand in S and of the receptor in R (CellPhoneDB-style), computed only for type pairs that are spatial neighbours in the k-nearest-neighbour graph of the centroids (k = 6) and only when ligand and receptor are detected in their type. The list of pairs was supplied by the user; no database is bundled. p-values: the cell-type labels were permuted within each sample (999 permutations, seed 0); p = (1 + permutations with a score >= observed) / (1 + permutations), so the smallest possible p is 1 / (1 + permutations); Benjamini-Hochberg FDR over all tested combinations. The unit of the p-values is the cell (labels exchangeable within a sample); it is not a test across samples or donors. A significant score is expression-based inference, not evidence of signalling. [1] [2]

Cell labels: celltype:celltypist (cell type). Original counts were used.

Parameters: pairs = CXCL12,CXCR4 CCL19,CCR7; k = 6; n_perm = 999; seed = 0.

Random seeds: seed = 0.

Unit of the p-values, as stated by the app: n = 1,100 cells, treated as exchangeable within their sample (label permutation); not a test across samples.

As reported by the app:

> 2 of 2 ligand–receptor pairs have both genes on the panel.

Caveats raised by the app:

- 999 permutations, seed 0; smallest possible p 0.001.

Software: Python 3.11.6; numpy 1.26.0; scipy 1.11.0; pandas 2.1.0; scikit-learn 1.3.0. Record prov-20260102-030405-lr0000 (computed, 1.5 s).

## References

1. Efremova M, Vento-Tormo M, Teichmann SA, Vento-Tormo R. CellPhoneDB: inferring cell-cell communication from combined expression of multi-subunit ligand-receptor complexes. Nat Protoc 15, 1484-1506 (2020).
2. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B 57, 289-300 (1995).

Entries marked [verify] could not be confirmed when the app's reference list was written: check them against the publication before citing.

## All records

| When | Analysis | Segmentation | Counts | Source | Seconds | Record |
|---|---|---|---|---|---|---|
| 2026-01-02 03:04:05 | Ligand-receptor scores | xenium | original | computed | 1.5 | prov-20260102-030405-lr0000 |
