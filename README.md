# OdorNet

**OdorNet** is a standardized molecular olfactory label dataset contributed and maintained by **the Lab of Digital Olfaction and Intelligent Equipment, College of Electronic Information and Optical Engineering, Nankai University (NKU-DOIE)**.

OdorNet was built using the SEA taxonomy framework, which integrates **Statistical co-occurrence**, **Expert correction**, and **AI-assisted semantic alignment** to organize heterogeneous molecular odor descriptors into a standardized hierarchical label system.

This GitHub repository provides the machine-learning-ready OdorNet data version: aligned molecule-level labels and the fixed stereochemistry-safe `7:2:1` train/validation/test split used for technical validation. Source-level records, raw curation files, taxonomy metadata, and files needed to reproduce the SEA processing are distributed through the Zenodo archive.

> **Maintainer statement**
> This project is contributed and maintained by the Lab of Digital Olfaction and Intelligent Equipment, College of Electronic Information and Optical Engineering, Nankai University (NKU-DOIE). External users are welcome to open GitHub Issues or Pull Requests for reproducibility questions, error reports, and suggested improvements. Final dataset curation and release decisions are maintained by the project maintainers.

## Dataset Description

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `data/processed/full_dataset.csv` | 8,892 | 13 | Full machine-learning label table with `SMILES` and 12 primary olfactory category labels. |
| `data/processed/dataset_train_aligned.csv` | 6,224 | 13 | Training split, approximately 70% of the consolidated molecules; unresolved labels are retained. |
| `data/processed/dataset_val_aligned.csv` | 1,778 | 13 | Validation split, approximately 20%; all 12 labels are explicit `0`/`1`. |
| `data/processed/dataset_test_aligned.csv` | 890 | 13 | Held-out test split, approximately 10%; all 12 labels are explicit `0`/`1`. |

The three split files are disjoint and their union matches `data/processed/full_dataset.csv`. The split is generated with canonical connectivity groups, so alternate SMILES and stereoisomers are not distributed across different splits. The training split preserves blank label cells for unresolved labels; validation and test contain no blank label cells.

To ensure a stable evaluation procedure, validation and test molecules are selected from complete-label connectivity groups. The split objective balances positive rates for all 12 labels within the complete-label subset; exact positive counts therefore differ by at most the unavoidable integer/group-size constraints.

The GitHub data directory intentionally keeps only the processed ML-ready tables. The `data/raw/`, `data/metadata/`, and `data/provenance/` directories are retained as placeholders and are populated only after restoring the Zenodo archive.

## Data Format

All processed CSV files contain `SMILES` and the 12 primary-category label columns:

```text
animalic&ambery, sweety&gourmand, floral, fruity&vegetable,
pungent&disagreeable, green&herbal, nutty, woody&mossy,
resinous&balsamic, cooked, odorless, spice
```

Label values are binary or unresolved:

- `1`: the category is present.
- `0`: the category is absent.
- blank cell: unresolved or missing label status.

The GitHub `full_dataset.csv` does not include a `Source` column. Source-level provenance is available from Zenodo.

Rows with all 12 released categories equal to `0` are not automatically
evidence that a molecule is odorless. They denote records whose source
descriptions were too vague, removed during descriptor mapping, or could not
be assigned to this SEA label space. They may be treated as explicit negatives
only for an analysis that states that modeling assumption. See
`data_dictionary.md` for the full encoding definition.

## Supporting Resources

`data/resources/` contains small public resources that support taxonomy and provenance inspection without exposing source-rich molecule tables:

- `fragrantica_notes_2026-08-22.csv`: 1,702 Fragrantica displayed term-category records from a dated PDF capture.
- `fragrantica_notes_2026-08-22_unique_terms.csv`: the 1,698 globally deduplicated terms from the same capture.
- `source_registry.csv`: Source IDs, proposed source times, record counts,
  descriptor coverage, DOI values, and source URLs used by the Zenodo release.

The complete nine-source catalog, including proposed times and Table 1 counts, is
available at `docs/source_catalog.md`.

The Fragrantica files are a transparent expert-review reference vocabulary only; they are not OdorNet targets and do not replace the released SEA mapping. See `data/resources/README.md` for collection scope and attribution notes.

See `data_dictionary.md` for field-level definitions, label names, and label counts.

## Dataset Versions and Missing-Label Policies

The aligned labels preserve unresolved cells so that different missing-label policies can be derived during modeling:

| Policy | Interpretation |
| --- | --- |
| `Drop` | Ignore blank labels during loss and metric computation. |
| `Union` | Treat blank labels as positive labels (`1`) when deriving a training target. |
| `Intersection` | Treat blank labels as negative labels (`0`) when deriving a training target. |

GitHub retains the unresolved base tables so that targets can be derived in
code. Zenodo additionally provides explicit `full_dataset_drop.csv`,
`full_dataset_union.csv`, and `full_dataset_intersection.csv`, each with an
`integration_policy` column.

## Loading the ML Dataset

```python
from pathlib import Path
import sys

root = Path(".").resolve()
sys.path.insert(0, str(root / "src"))

from odornet.datasets import LABEL_COLUMNS, label_frame, load_odornet

full_df = load_odornet("full", root=root)
train_df = load_odornet("train", root=root)
validation_df = load_odornet("val", root=root)
test_df = load_odornet("test", root=root)

train_labels = label_frame(train_df)

drop_targets = train_labels
drop_mask = train_labels.notna()

union_targets = train_labels.fillna(1.0)
intersection_targets = train_labels.fillna(0.0)

print(full_df.shape, train_df.shape, validation_df.shape, test_df.shape)
print(LABEL_COLUMNS)
print("Drop valid entries:", int(drop_mask.to_numpy().sum()))
print("Union positives:", union_targets.sum().sort_values(ascending=False).head())
print("Intersection positives:", intersection_targets.sum().sort_values(ascending=False).head())
```

`load_odornet()` reads the aligned release files without changing blank cells. The three modeling policies are derived after loading: `Drop` keeps blank cells masked out, `Union` converts blank cells to positive targets, and `Intersection` converts blank cells to negative targets. A runnable example is provided in `examples/demo_load_odornet.py`.

## Baseline Training

Baseline training does **not** require the Zenodo archive. The baseline notebook uses the processed train, validation, and test CSV files already included in this GitHub repository:

- `notebooks/odornet_baseline_training.ipynb`: fixed-split baseline training following the reference notebook's MolFormer fine-tuning and simple GCN/GNN baselines under Drop, Union, and Intersection missing-label policies.

## Technical Validation Notebook

`notebooks/technical_validation.ipynb` is the single public notebook for the
technical-validation workflow. It contains:

- SEA construction from descriptor statistics, expert correction, and
  AI-assisted semantic alignment;
- label-space consistency, coverage, and source-label deletion audits;
- RDKit cleaning and canonical structure checks;
- connectivity-aware `7:2:1` split validation with no validation/test NaN cells;
- raw multi-dataset comparisons and Drop/Union/Intersection training;
- publication-oriented result and training-curve visualizations.

The notebook exposes separate switches for the expensive training stages and
reuses cached result summaries when they are present. Training details are
documented in the notebook and in `docs/reviewer_revision.md`.

## Reproducing SEA Processing and Split Construction

The SEA processing notebook and split-strategy notebook require source-level data and taxonomy metadata that are distributed through Zenodo, not GitHub:

- `notebooks/odornet_sea_pipeline.ipynb`: source loading, descriptor co-occurrence statistics, expert-corrected SEA mapping, Double-Drop label generation, source merge, and release-label verification.
- `notebooks/odornet_split_strategy.ipynb`: historical split-design notes retained for reference.

To reproduce these notebooks from a clean clone, restore the Zenodo archive over the repository data directory:

```bash
git clone https://github.com/NKU-DOIE/OdorNet.git
cd OdorNet
python scripts/download_zenodo_release.py --extract --output-dir zenodo_release
cp -a zenodo_release/<release-directory>/data/. data/
```

After this overlay, `data/raw/`, `data/metadata/`, `data/provenance/`, and the source-rich Zenodo version of `data/processed/full_dataset.csv` are available locally, and the technical-validation notebook can be executed.

## Reviewer Validation

Reviewer-requested taxonomy audits, controlled GNN/MolFormer comparisons, and randomized-hierarchy experiments are documented in `docs/reviewer_revision.md`. The corresponding implementation modules and command-line runner are:

```bash
python scripts/run_reviewer_evaluations.py --stage all
```

The script writes public summary tables and figures to `results/reviewer_revision/`, while model checkpoints and epoch logs remain under ignored `outputs/reviewer_revision/`. The unified notebook is the recommended entry point; the GS_lf comparison expects the locally supplied `data/raw/gs_lf.csv` file and does not publish that input through GitHub.

## Zenodo Data Archive

The Zenodo archive contains the source-rich data release required for curation inspection and deterministic SEA/split reproduction.

- Concept DOI (resolves to the newest published version): `10.5281/zenodo.19838455`
- Current record DOI before the 1.1.0 publication: `10.5281/zenodo.19838456`
- Record URL: `https://zenodo.org/records/19838455`
- Download URL: provided by the newest published Zenodo version
- Community: `https://zenodo.org/communities/nku-logic/`

The archive includes processed tables, row-wise provenance files, metadata, raw source-level metadata, a data-only README, release notes, and SHA-256 checksums. The helper script downloads the archive, checks the archive SHA-256, extracts it, and verifies every file listed in `checksums_sha256.txt`.

## Repository Structure

```text
.
├── data/
│   ├── processed/
│   │   ├── full_dataset.csv
│   │   ├── dataset_train_aligned.csv
│   │   ├── dataset_val_aligned.csv
│   │   └── dataset_test_aligned.csv
│   ├── resources/
│   │   ├── fragrantica_notes_2026-08-22.csv
│   │   ├── fragrantica_notes_2026-08-22_unique_terms.csv
│   │   └── source_registry.csv
│   ├── raw/             # populated from Zenodo when reproducing SEA processing
│   ├── metadata/        # populated from Zenodo when reproducing SEA processing
│   └── provenance/      # populated from Zenodo when inspecting source records
├── examples/
│   └── demo_load_odornet.py
├── scripts/
│   ├── download_zenodo_release.py
│   └── build_zenodo_release.py
├── notebooks/
│   ├── odornet_baseline_training.ipynb
│   └── technical_validation.ipynb
├── src/
│   └── odornet/
├── README.md
├── data_dictionary.md
├── docs/
│   ├── reviewer_revision.md
│   └── source_catalog.md
├── results/
│   └── reviewer_revision/
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── LICENSE-MIT
├── LICENSE-CC-BY-4.0
└── requirements.txt
```

## Citation

If you use OdorNet, please cite the dataset repository and the associated paper once available.

For the current release, use `CITATION.cff`. The Zenodo concept DOI is
`10.5281/zenodo.19838455`; journal and final publication metadata are to be
confirmed.

## License

This repository uses separate licenses:

- License summary: `LICENSE`.
- Code, examples, notebooks, and documentation: MIT License (`LICENSE-MIT`).
- Dataset files under `data/processed/`: Creative Commons Attribution 4.0 International (`LICENSE-CC-BY-4.0`).
- Source-rich data and metadata restored from Zenodo: Creative Commons Attribution 4.0 International (`LICENSE-CC-BY-4.0`).

## Original Data Sources

OdorNet integrates and standardizes public molecular odor labels from literature and open databases. Users should cite OdorNet and remain responsible for citing original data sources when source-specific records, raw labels, or historical interpretations are used in downstream work.

## Contributing and Maintenance

This repository is maintained by **the Lab of Digital Olfaction and Intelligent Equipment, College of Electronic Information and Optical Engineering, Nankai University (NKU-DOIE)**.

We welcome community feedback through GitHub Issues and Pull Requests, especially for:

- reproducibility questions;
- documentation improvements;
- suspected data errors;
- source-provenance corrections;
- suggestions for benchmark reuse.

For dataset integrity, changes to released data files are reviewed by the maintainers before acceptance. Contributors should not directly modify released data tables without explaining the motivation, affected records, and validation procedure.

## Contact

Please use GitHub Issues for questions, corrections, and reproducibility reports.

Maintainer: **the Lab of Digital Olfaction and Intelligent Equipment, College of Electronic Information and Optical Engineering, Nankai University (NKU-DOIE)**

Contact: **yangliyuanyly@mail.nankai.edu.cn**
