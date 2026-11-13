# Methods


## Data

1 sample(s) were loaded (S0: sectionA, 1,200 cells), 1,200 cells and 300 genes in the primary sample's panel.

## Differential expression (cells as observations)

Differential expression between two groups of cells of the Xenium segmentation (A: T cell, n = 200 cells; B: all other cells, n = 960 cells) was tested for every measured gene (300 genes) with the two-sided Wilcoxon rank-sum test on log1p(counts per 10,000) and p-values were adjusted with the Benjamini-Hochberg procedure over the tested genes. Cells were treated as independent observations; this test does not use the sample design, so with cells from few samples its p-values are too small (pseudo-replication) and rank genes rather than establish significance. A gene counted as detected in a cell with a count of at least one transcript. The leak gate (a model of transcripts mis-assigned from neighbouring cell types, with gene ownership per cell type) was requested; what it did is in the app's report below (it is computed only when group A is one annotated cell type). [1]

Cell labels: celltype:celltypist (cell type). SPLIT-corrected counts were used; 40 cells without SPLIT-corrected counts were left out.

Parameters: group_a.kind = celltype; group_a.value = T cell; color_by = cell_type; options.leak_gate = yes.

Unit of the p-values, as stated by the app: n = 1,160 cells, treated as independent (they are not, when the groups come from few samples: the p-values are then far too small).

As reported by the app:

> T cell (A, 200 cells) vs all other cells (B, 960 cells): 300 genes, 25 with FDR < 0.05 (SPLIT-corrected counts, 0.4 s).
> Unit of the p-values: n = 1,160 cells, treated as independent (they are not, when the groups come from few samples: the p-values are then far too small).

Caveats raised by the app:

- Leak gate: 12 of 300 genes flagged.
- 40 cells have no SPLIT-corrected counts and were left out.

Software: Python 3.11.6; numpy 1.26.0; scipy 1.11.0; pandas 2.1.0; scikit-learn 1.3.0. Record prov-20260102-030405-de0000 (computed, 1.5 s).

## References

1. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B 57, 289-300 (1995).

Entries marked [verify] could not be confirmed when the app's reference list was written: check them against the publication before citing.

## All records

| When | Analysis | Segmentation | Counts | Source | Seconds | Record |
|---|---|---|---|---|---|---|
| 2026-01-02 03:04:05 | Differential expression (cells as observations) | xenium | SPLIT-corrected | computed | 1.5 | prov-20260102-030405-de0000 |
