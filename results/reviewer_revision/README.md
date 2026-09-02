# Technical-Validation Result Summaries

This directory tracks only lightweight, publication-facing summaries used by
`notebooks/technical_validation.ipynb`. Checkpoints, epoch logs, source-derived
external datasets, generated splits, and legacy threshold-optimized diagnostic
results are deliberately excluded.

## Included Results

- `smiles_preprocessing_summary.json` and
  `smiles_preprocessing_by_source.csv`: RDKit source-processing audit.
- `label_coverage_summary.csv`, `sea_label_deletion_stability_summary.json`,
  and `semantic_consistency_summary.json`: SEA consistency and stability
  summaries.
- `raw_multi_dataset_comparison/`: fixed-threshold native-label comparison
  metrics and split audits.
- `sea_integration_modes/`: fixed-threshold Drop, Union, and Intersection
  metrics plus the released split audits.
- `figures/`: bar charts generated from the two fixed-threshold result tables.

All reported F1 values are Macro F1 at the fixed threshold 0.5. The raw
multi-dataset scores use different native label spaces and are therefore
training-protocol comparisons, not a common-label-space leaderboard.
