# Methods


## Data

1 sample(s) were loaded (S0: sectionA, 1,200 cells), 1,200 cells and 300 genes in the primary sample's panel.

## Pseudobulk differential expression

Pseudobulk differential expression of the population 'T cell' between the two conditions of the sample design: the counts of the population's cells were summed per donor and condition (A: 3 donors; B: 3 donors); a sample with fewer than 10 cells of the population was dropped and a gene with a total count below 10 was not tested. Model: DESeq2 (Wald test), design ~ donor + condition; log2 fold changes are A / B (unshrunken maximum-likelihood estimates; shrunken estimates: apeglm); adjusted p-values are those of the fitting package (Benjamini-Hochberg). 280 genes were tested, 20 were not (low count), and 0 have no adjusted p-value. Original (uncorrected) counts were summed; SPLIT-corrected counts are not used for count models. [1] [2]

Cell labels: celltype:celltypist (cell type). Original counts were used. Sample design as set in the app: condition A ('treated'): S0 (donor d0), S1 (donor d1), S2 (donor d2); condition B ('control'): S3 (donor d0), S4 (donor d1), S5 (donor d2). Donor labels were given and at least one donor occurs in both conditions (paired).

Parameters: population = T cell; min_cells = 10; min_count = 10; shrink = yes; engine = auto.

Unit of the p-values, as stated by the app: n = 3 donors per condition (the donors: not their sections, not the cells).

As reported by the app:

> Pseudobulk (T cell): 280 genes tested, 4 with FDR < 0.05 (3.2 s).

Software: Python 3.11.6; numpy 1.26.0; R 4.3.1 (2023-06-16); DESeq2 1.40.2 (R); edgeR not found (R). Record prov-20260102-030405-pseudo (computed, 1.5 s).

## References

1. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B 57, 289-300 (1995).
2. Love MI, Huber W, Anders S. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biol 15, 550 (2014).

Entries marked [verify] could not be confirmed when the app's reference list was written: check them against the publication before citing.

## All records

| When | Analysis | Segmentation | Counts | Source | Seconds | Record |
|---|---|---|---|---|---|---|
| 2026-01-02 03:04:05 | Pseudobulk differential expression | xenium | original | computed | 1.5 | prov-20260102-030405-pseudo |
