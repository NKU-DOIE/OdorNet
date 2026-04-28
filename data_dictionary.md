# OdorNet Data Dictionary

This data dictionary describes the machine-learning-ready OdorNet files included in the GitHub repository:

- `data/processed/full_dataset.csv`
- `data/processed/dataset_train_aligned.csv`
- `data/processed/dataset_test_aligned.csv`

Source-level records, raw curation files, taxonomy metadata, and the source-rich version of `full_dataset.csv` are distributed through Zenodo (`10.5281/zenodo.19838456`).

## Table-Level Summary

| File | Rows | Columns | Duplicate SMILES | Notes |
| --- | ---: | ---: | ---: | --- |
| `data/processed/full_dataset.csv` | 8,892 | 13 | 0 | Full machine-learning label table without source provenance. |
| `data/processed/dataset_train_aligned.csv` | 7,114 | 13 | 0 | Training split; contains blank label cells. |
| `data/processed/dataset_test_aligned.csv` | 1,778 | 13 | 0 | Held-out validation/evaluation split; no blank label cells observed. |

The train and validation SMILES sets are disjoint, and their union equals `data/processed/full_dataset.csv`.

## General Label Encoding

| Value | Meaning |
| --- | --- |
| `1` or `1.0` | The molecule is annotated with the odor category. |
| `0` or `0.0` | The molecule is not annotated with the odor category. |
| Blank cell | Missing, unresolved, or policy-dependent label status. |

The `data/processed/full_dataset.csv` file stores numeric labels as `0.0`/`1.0`, while the train/validation split files store labels as `0`/`1`. These encodings should be treated equivalently after numeric parsing.

## Columns

| Column | Type | Files | Description |
| --- | --- | --- | --- |
| `SMILES` | string | all processed CSVs | Molecular structure represented as a SMILES string. No duplicate SMILES were observed. |
| `animalic&ambery` | binary / blank | all processed CSVs | Primary olfactory category label for animalic and ambery odors. |
| `sweety&gourmand` | binary / blank | all processed CSVs | Primary olfactory category label for sweet and gourmand odors. |
| `floral` | binary / blank | all processed CSVs | Primary olfactory category label for floral odors. |
| `fruity&vegetable` | binary / blank | all processed CSVs | Primary olfactory category label for fruity and vegetable-like odors. |
| `pungent&disagreetable` | binary / blank | all processed CSVs | Primary olfactory category label for pungent and disagreeable odors. The column name is kept exactly as released in the CSV files. |
| `green&herbal` | binary / blank | all processed CSVs | Primary olfactory category label for green and herbal odors. |
| `nutty` | binary / blank | all processed CSVs | Primary olfactory category label for nutty odors. |
| `woody&mossy` | binary / blank | all processed CSVs | Primary olfactory category label for woody and mossy odors. |
| `resinous&balsamic` | binary / blank | all processed CSVs | Primary olfactory category label for resinous and balsamic odors. |
| `cooked` | binary / blank | all processed CSVs | Primary olfactory category label for cooked odors. |
| `odorless` | binary / blank | all processed CSVs | Label indicating odorless molecules. |
| `spice` | binary / blank | all processed CSVs | Primary olfactory category label for spice odors. |

## Source Provenance

The GitHub version of `data/processed/full_dataset.csv` intentionally omits the `Source` column. To inspect source-level records or reproduce the SEA processing pipeline, restore the Zenodo archive over the repository data directory:

```bash
python scripts/download_zenodo_release.py --extract --output-dir zenodo_release
cp -a zenodo_release/OdorNet_v1.0.0/data/. data/
```

After this overlay, source-rich files are available under:

- `data/provenance/source_records.jsonl`
- `data/provenance/source_records.csv`
- `data/raw/merged_8892_cleaned_251230.pkl`
- `data/metadata/`

## Fields Not Present in the GitHub ML Tables

The GitHub processed CSV files do not include the following fields:

- `Source`
- `CAS`
- `molecule_id`
- `raw_label` as a top-level column
- `standardized_label` as a top-level column
- `primary_category` as a single categorical column
- `secondary_category`
- `split`
- explicit `Union`, `Intersection`, or `Drop` columns

## Positive Label Counts

Positive counts in `data/processed/full_dataset.csv`:

| Label | Positive count |
| --- | ---: |
| `green&herbal` | 2,809 |
| `fruity&vegetable` | 1,711 |
| `floral` | 1,619 |
| `odorless` | 1,359 |
| `pungent&disagreetable` | 1,306 |
| `sweety&gourmand` | 1,300 |
| `woody&mossy` | 1,091 |
| `spice` | 662 |
| `resinous&balsamic` | 515 |
| `animalic&ambery` | 442 |
| `nutty` | 327 |
| `cooked` | 301 |

## Missing Label Counts

Missing counts in `data/processed/full_dataset.csv`:

| Label | Blank cells |
| --- | ---: |
| `fruity&vegetable` | 2,635 |
| `sweety&gourmand` | 1,945 |
| `pungent&disagreetable` | 1,786 |
| `animalic&ambery` | 1,550 |
| `green&herbal` | 1,544 |
| `woody&mossy` | 1,490 |
| `spice` | 1,328 |
| `cooked` | 1,101 |
| `floral` | 1,066 |
| `nutty` | 590 |
| `resinous&balsamic` | 542 |
| `odorless` | 228 |
