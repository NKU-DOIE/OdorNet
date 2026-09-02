# OdorNet Data Dictionary

This document defines the processed GitHub tables and the provenance-rich
tables included in the OdorNet Zenodo 1.1.0 package. Canonical source IDs,
references, persistent identifiers, time ranges, and source relationships are
listed in `docs/source_catalog.md` and `data/resources/source_registry.csv`.

## Processed GitHub Tables

| File | Rows | Fields | Purpose |
| --- | ---: | ---: | --- |
| `data/processed/full_dataset.csv` | 8,892 | `SMILES` plus 12 labels | Unresolved SEA base table. |
| `data/processed/dataset_train_aligned.csv` | 6,224 | `SMILES` plus 12 labels | 70% training partition; unresolved labels are retained. |
| `data/processed/dataset_val_aligned.csv` | 1,778 | `SMILES` plus 12 labels | 20% validation partition; every label is explicit. |
| `data/processed/dataset_test_aligned.csv` | 890 | `SMILES` plus 12 labels | 10% held-out test partition; every label is explicit. |
| `data/processed/split_manifest.json` | n/a | JSON metadata | Split ratios, row counts, missing-label counts, group-aware strategy, seed, and checksums. |

The three split files are pairwise disjoint and their union equals
`full_dataset.csv`. The split is made by canonical connectivity group, so the
same connectivity (including alternate SMILES notations and stereoisomers)
cannot appear in more than one partition. Validation and test contain zero
missing label cells. The training table contains 15,805 unresolved cells so
that the three integration policies can be derived without changing the
released evidence table.

## Column Definitions

| Column | Type | Meaning |
| --- | --- | --- |
| `SMILES` | string | Standardized isomeric molecular SMILES used as the molecule identifier in the processed release. |
| `animalic&ambery` | binary / blank | SEA primary category: animalic and ambery. |
| `sweety&gourmand` | binary / blank | SEA primary category: sweet and gourmand. |
| `floral` | binary / blank | SEA primary category: floral. |
| `fruity&vegetable` | binary / blank | SEA primary category: fruity and vegetable-like. |
| `pungent&disagreeable` | binary / blank | SEA primary category: pungent and disagreeable. |
| `green&herbal` | binary / blank | SEA primary category: green and herbal. |
| `nutty` | binary / blank | SEA primary category: nutty. |
| `woody&mossy` | binary / blank | SEA primary category: woody and mossy. |
| `resinous&balsamic` | binary / blank | SEA primary category: resinous and balsamic. |
| `cooked` | binary / blank | SEA primary category: cooked. |
| `odorless` | binary / blank | SEA primary category: odorless. |
| `spice` | binary / blank | SEA primary category: spice. |

Each label has the following encoding:

| Stored value | Meaning |
| --- | --- |
| `1` or `1.0` | The category is supported by the resolved SEA annotation. |
| `0` or `0.0` | The category is explicitly absent after resolution. |
| blank / `NA` | The category is unresolved in the base table; it is not a negative label. |

A row whose 12 category values are all `0` is **not automatically an
odorless molecule**. It indicates that source annotations were too vague,
deleted during mapping, or could not be assigned to the current 12-category
SEA label space. Such rows can serve as explicit negatives for the released
label space only under an analysis that makes that choice explicit; they must
not be interpreted as chemical evidence of an absence of odor.

## Missing-Label Integration Policies

The base table retains the source-derived unresolved state. The policy is a
modeling decision, not an additional annotation:

| Policy | Treatment of blank label cells | Use in release |
| --- | --- | --- |
| `Drop` | Mask unresolved cells out of loss and metric calculations. | Native evidence-preserving mode. |
| `Union` | Convert unresolved cells to `1`. | Permissive derived target. |
| `Intersection` | Convert unresolved cells to `0`. | Conservative derived target. |

The Zenodo package explicitly supplies `full_dataset_drop.csv`,
`full_dataset_union.csv`, and `full_dataset_intersection.csv`. Each has an
`integration_policy` column. GitHub provides the unresolved base table and
the fixed 7:2:1 release split; the same derivation is available through
`odornet.datasets`.

## Zenodo Provenance Tables

Zenodo contains source-rich data unavailable in GitHub because it preserves
the supplied source-level records. The principal files are:

| File | Granularity | Description |
| --- | --- | --- |
| `data/provenance/source_records.csv` | one source annotation per row | Flat, spreadsheet-friendly provenance table. |
| `data/provenance/source_records.jsonl` | one source annotation per line | Equivalent provenance records for streaming tools. |
| `data/provenance/source_registry.csv` | one of nine sources per row | Canonical identifiers and source-level metadata. |
| `data/raw/merged_8892_cleaned_251230.pkl` | one consolidated molecule per row | Source-list-preserving table used by SEA preparation. |

The Zenodo `data/processed/full_dataset*.csv` files add a `Source` column not
present in GitHub. It is a JSON array of the enriched source records supporting
the molecule and should be parsed as structured JSON, not as a delimiter-split
text field. The Zenodo policy tables additionally carry `integration_policy`.

### `source_records.csv` fields

| Field | Meaning |
| --- | --- |
| `SMILES` | Consolidated standardized molecule SMILES. |
| `Source` / `source_id` | Canonical source ID used in the manuscript and release. |
| `legacy_source_id` | Pre-release source label found in the supplied working table. |
| `source_record_id` | Deterministic ID assigned to the source record during package construction. |
| `original_identifier` | Source-provided structure identifier. In this snapshot it is the original SMILES. |
| `original_identifier_type` | Identifier type, fixed to `Original_SMILES` because source-native row IDs were not supplied. |
| `Original_SMILES` | Structure string retained from the source record. |
| `Original_IUPACname` | Source-provided IUPAC name when available. |
| `Original_Labels` / `raw_label` | Unchanged source label text. |
| `Pre_Processed_Labels` | Parsed source descriptors before the final normalization stage. |
| `Processed_Labels` / `processed_label` | Normalized descriptor list supplied to SEA. |
| `source_display_name`, `source_type`, `source_reference` | Human-readable identity and citation. |
| `source_doi`, `source_doi_status`, `source_url` | Persistent identifier status and source website. |
| `time_range` | Publication date or website/snapshot period represented by this release. |
| `raw_record_count`, `unique_molecule_count`, `descriptor_coverage` | Source summary counts used in the catalog. |
| `annotation_procedure` | How the labels were captured in the supplied source snapshot. |
| `source_relationships`, `provenance_notes` | Known reuse, aggregation, or source-specific caveats. |

### `source_registry.csv` fields

`source_registry.csv` contains the source-level metadata fields above, except
molecule-specific annotation fields. It has exactly nine rows, one for each
canonical source. The canonical mapping is intentionally stable:

| Canonical source ID | Legacy source ID |
| --- | --- |
| `arctander` | `arctander_1960` |
| `aromadb` | `aromadb` |
| `flavordb` | `flavordb` |
| `flavornet` | `flavornet` |
| `TGSC` | `goodscents` |
| `IFRA` | `ifra_2019` |
| `Leffingwell` | `leffingwell` |
| `SMILES_to_smell` | `sharma_2021a` |
| `OlfactionBase` | `sharma_2021b` |

`OlfactionBase` is associated with the `SMILES_to_smell` publication and the
relevant records include its odorless/PubChem compilation. Beryllium-related
records in the source snapshot are documented as originating from that
compilation; the provenance record does not make an independent toxicity
claim.

## Positive and Missing Counts

The full unresolved base table contains these positive and missing values:

| Label | Positive count | Blank count |
| --- | ---: | ---: |
| `green&herbal` | 2,809 | 1,544 |
| `fruity&vegetable` | 1,711 | 2,635 |
| `floral` | 1,619 | 1,066 |
| `odorless` | 1,359 | 228 |
| `pungent&disagreeable` | 1,306 | 1,786 |
| `sweety&gourmand` | 1,300 | 1,945 |
| `woody&mossy` | 1,091 | 1,490 |
| `spice` | 662 | 1,328 |
| `resinous&balsamic` | 515 | 542 |
| `animalic&ambery` | 442 | 1,550 |
| `nutty` | 327 | 590 |
| `cooked` | 301 | 1,101 |

## Reproducibility References

`notebooks/technical_validation.ipynb` is the single public technical
validation notebook. It documents RDKit cleaning, SEA construction,
label-space checks, the connectivity-safe split audit, common training
protocol, raw multi-dataset comparison, policy comparison, and plotting.
The `split_manifest.json` and SHA-256 package checksums are the release
integrity records.
