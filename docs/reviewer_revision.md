# Technical Validation and Reviewer Revision

The public technical-validation record is
`notebooks/technical_validation.ipynb`. It is the single notebook that joins
the source audit, SEA construction, label-space checks, released 7:2:1 split
audit, fixed-protocol comparison experiments, and visualizations. Source-level
inputs are restored from the Zenodo package before running the SEA sections.

## Source Processing

The source audit reads `data/raw/merged_8892_cleaned_251230.pkl`, expands its
nested source records, parses each `Original_SMILES` with RDKit, sanitizes it,
and canonicalizes it as an isomeric SMILES. The supplied snapshot contains
23,247 source records. There are zero blank SMILES inputs, parser failures,
sanitization failures, or canonicalization failures; therefore no record is
removed at the observable RDKit stage. Deduplication retains 8,892 unique
cleaned isomeric SMILES, matching the released full table.

## SEA Reproducibility

SEA combines three explicit layers:

1. Statistical co-occurrence: the notebook saves `C = X.T @ X` and the
   directed conditional-probability matrix `P(B|A)`, using one binary
   descriptor transaction per consolidated molecule. Display edges require
   descriptor frequency of at least 30 and `P(B|A) >= 0.45`.
2. Expert correction: `final_specialist_label_mapping.json` contributes 190
   descriptor-category relations across eight primary categories.
3. AI-assisted alignment: English definitions in
   `olfactory_data_translated.json` were used in the historical nine-vote
   protocol. Assignments with at least eight votes are strong, three to seven
   are weak, and labels below three votes are `not any type`.

The notebook then audits label coverage, source-label deletion stability, and
semantic consistency. The cached semantic audit embeds 490 descriptors with
English definitions; its within-primary macro mean cosine similarity is
0.4104 versus 0.3869 between primary categories. These are consistency
measurements, not external accuracy estimates.

## Released Split

`data/processed/split_manifest.json` documents the published 7:2:1 split:
6,224 training rows, 1,778 validation rows, and 890 test rows. It is grouped
by canonical molecular connectivity. Pairwise checks between all three
partitions find zero stored-SMILES, canonical-isomeric-SMILES, and canonical
connectivity overlaps. Validation and test each have zero unresolved label
cells; training retains 15,805 unresolved cells for the policy comparison.

## Shared Training Protocol

The raw multi-dataset and SEA integration-mode comparisons use identical
training rules: seed 959, 30 epochs, batch size 48, class-weighted binary
cross-entropy with logits, training-only shuffling, a fixed threshold of 0.5,
and best-checkpoint selection by validation Macro F1. No learning-rate
scheduler, gradient clipping, mixed precision, threshold tuning, or early
stopping is used. MolFormer uses the fine-tuned 256-token encoder and
`768-512-384-256` MLP; the GNN uses two 256-dimensional GCNConv layers and a
`256-128-output` classifier. Complete architecture and optimizer parameters
are stated in the notebook.

## Fixed-Threshold Results

The raw comparison keeps each external dataset's native filtered label space;
therefore its scores are protocol-matched but are not a common-label-space
benchmark.

| Dataset | Labels | GNN test Macro F1 | MolFormer test Macro F1 |
| --- | ---: | ---: | ---: |
| OdorNet SEA | 12 | 0.3248 | 0.4228 |
| OdorNet raw labels | 84 | 0.1476 | 0.2314 |
| GS_LF | 100 | 0.1549 | 0.2084 |
| SMILES_to_smell | 119 | 0.1257 | 0.1622 |
| Arctander | 50 | 0.1288 | 0.1937 |

| SEA policy | GNN test Macro F1 | MolFormer test Macro F1 |
| --- | ---: | ---: |
| Drop | 0.3276 | 0.4228 |
| Union | 0.2753 | 0.3620 |
| Intersection | 0.2355 | 0.3703 |

The historical threshold-optimized controlled and hierarchy-randomization
outputs are retained locally as diagnostic material but are not presented as
the fixed-threshold release benchmark, and are not included in the unified
notebook.

## Provenance and Vocabulary

`data/resources/fragrantica_notes_2026-08-22.csv` contains 1,702 displayed
term-category records and 1,698 globally deduplicated terms. It is a
transparent expert-review vocabulary, not an OdorNet target mapping.

The nine-source catalog and provenance fields are documented in
`docs/source_catalog.md` and `data_dictionary.md`. Zenodo preserves each raw
source label alongside its original molecule information, Source ID, DOI, and
source URL. Source dates, counts, and descriptor coverage are reported once in
the catalog.
