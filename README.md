# OdorNet

**OdorNet** is a standardized molecular olfactory label dataset contributed and maintained by **the Lab of Digital Olfaction and Intelligent Equipment, College of Electronic Information and Optical Engineering, Nankai University (NKU-DOIE)**.

OdorNet was built using the SEA taxonomy framework, which integrates **Statistical co-occurrence**, **Expert correction**, and **AI-assisted semantic alignment** to organize heterogeneous molecular odor descriptors into a standardized hierarchical label system.

This GitHub repository provides the machine-learning-ready OdorNet data version: aligned molecule-level labels and the fixed train/validation split used for baseline training and evaluation. Source-level records, raw curation files, taxonomy metadata, and files needed to reproduce the SEA processing and train/validation split are distributed through the Zenodo archive.

> **Maintainer statement**
> This project is contributed and maintained by the Lab of Digital Olfaction and Intelligent Equipment, College of Electronic Information and Optical Engineering, Nankai University (NKU-DOIE). External users are welcome to open GitHub Issues or Pull Requests for reproducibility questions, error reports, and suggested improvements. Final dataset curation and release decisions are maintained by the project maintainers.

## Dataset Description

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `data/processed/full_dataset.csv` | 8,892 | 13 | Full machine-learning label table with `SMILES` and 12 primary olfactory category labels. |
| `data/processed/dataset_train_aligned.csv` | 7,114 | 13 | Training split used for fixed-split baseline training. |
| `data/processed/dataset_test_aligned.csv` | 1,778 | 13 | Held-out validation/evaluation split used for fixed-split baseline evaluation. |

The train and validation files have no overlapping SMILES strings, and their union matches `data/processed/full_dataset.csv`. The training split preserves blank label cells for unresolved labels; the validation split contains only explicit `0` or `1` labels.

To ensure a stable validation procedure, the released split uses a constrained strategy: the validation set is sampled only from molecules with complete labels, while molecules with any unresolved label are kept in the training set. Multi-label stratification is used on the complete-label subset to keep positive and negative label ratios close between train and validation. This design removes NaN labels from validation metrics but may introduce random-selection bias.

The GitHub data directory intentionally keeps only the processed ML-ready tables. The `data/raw/`, `data/metadata/`, and `data/provenance/` directories are retained as placeholders and are populated only after restoring the Zenodo archive.

## Data Format

All processed CSV files contain `SMILES` and the 12 primary-category label columns:

```text
animalic&ambery, sweety&gourmand, floral, fruity&vegetable,
pungent&disagreetable, green&herbal, nutty, woody&mossy,
resinous&balsamic, cooked, odorless, spice
```

Label values are binary or unresolved:

- `1`: the category is present.
- `0`: the category is absent.
- blank cell: unresolved or missing label status.

The GitHub `full_dataset.csv` does not include a `Source` column. Source-level provenance is available from Zenodo.

See `data_dictionary.md` for field-level definitions, label names, and label counts.

## Dataset Versions and Missing-Label Policies

The aligned labels preserve unresolved cells so that different missing-label policies can be derived during modeling:

| Policy | Interpretation |
| --- | --- |
| `Drop` | Ignore blank labels during loss and metric computation. |
| `Union` | Treat blank labels as positive labels (`1`) when deriving a training target. |
| `Intersection` | Treat blank labels as negative labels (`0`) when deriving a training target. |

Separate Union, Intersection, and Drop CSV files are not included. Users can derive these versions from the aligned release files according to the policy definitions above.

## Loading the ML Dataset

```python
from pathlib import Path
import sys

root = Path(".").resolve()
sys.path.insert(0, str(root / "src"))

from odornet.datasets import LABEL_COLUMNS, label_frame, load_odornet

full_df = load_odornet("full", root=root)
train_df = load_odornet("train", root=root)
validation_df = load_odornet("test", root=root)

train_labels = label_frame(train_df)

drop_targets = train_labels
drop_mask = train_labels.notna()

union_targets = train_labels.fillna(1.0)
intersection_targets = train_labels.fillna(0.0)

print(full_df.shape, train_df.shape, validation_df.shape)
print(LABEL_COLUMNS)
print("Drop valid entries:", int(drop_mask.to_numpy().sum()))
print("Union positives:", union_targets.sum().sort_values(ascending=False).head())
print("Intersection positives:", intersection_targets.sum().sort_values(ascending=False).head())
```

`load_odornet()` reads the aligned release files without changing blank cells. The three modeling policies are derived after loading: `Drop` keeps blank cells masked out, `Union` converts blank cells to positive targets, and `Intersection` converts blank cells to negative targets. A runnable example is provided in `examples/demo_load_odornet.py`.

## Baseline Training

Baseline validation does **not** require the Zenodo archive. The baseline notebook uses the processed train and validation CSV files already included in this GitHub repository:

- `notebooks/odornet_baseline_training.ipynb`: fixed-split baseline training following the reference notebook's MolFormer fine-tuning and simple GCN/GNN baselines under Drop, Union, and Intersection missing-label policies.

## Reproducing SEA Processing and Split Construction

The SEA processing notebook and split-strategy notebook require source-level data and taxonomy metadata that are distributed through Zenodo, not GitHub:

- `notebooks/odornet_sea_pipeline.ipynb`: source loading, descriptor co-occurrence statistics, expert-corrected SEA mapping, Double-Drop label generation, source merge, and release-label verification.
- `notebooks/odornet_split_strategy.ipynb`: train/validation split design, no-NaN validation constraint, perfect/imperfect molecule separation, and label-balance audit.

To reproduce these notebooks from a clean clone, restore the Zenodo archive over the repository data directory:

```bash
git clone https://github.com/NKU-DOIE/OdorNet.git
cd OdorNet
python scripts/download_zenodo_release.py --extract --output-dir zenodo_release
cp -a zenodo_release/OdorNet_v1.0.0/data/. data/
```

After this overlay, `data/raw/`, `data/metadata/`, `data/provenance/`, and the source-rich Zenodo version of `data/processed/full_dataset.csv` are available locally, and the SEA and split notebooks can be executed.

## Zenodo Data Archive

The Zenodo archive contains the source-rich data release required for curation inspection and deterministic SEA/split reproduction.

- DOI: `10.5281/zenodo.19838456`
- Record URL: `https://zenodo.org/records/19838456`
- Download URL: `https://zenodo.org/records/19838456/files/OdorNet_v1.0.0.zip?download=1`
- Community: `https://zenodo.org/communities/nku-logic/`

The archive includes processed tables, row-wise provenance files, metadata, raw source-level metadata, a data-only README, release notes, and SHA-256 checksums. The helper script downloads the archive, checks the archive SHA-256, extracts it, and verifies every file listed in `checksums_sha256.txt`.

## Repository Structure

```text
.
├── data/
│   ├── processed/
│   │   ├── full_dataset.csv
│   │   ├── dataset_train_aligned.csv
│   │   └── dataset_test_aligned.csv
│   ├── raw/             # populated from Zenodo when reproducing SEA processing
│   ├── metadata/        # populated from Zenodo when reproducing SEA processing
│   └── provenance/      # populated from Zenodo when inspecting source records
├── examples/
│   └── demo_load_odornet.py
├── scripts/
│   └── download_zenodo_release.py
├── notebooks/
│   ├── odornet_sea_pipeline.ipynb
│   ├── odornet_split_strategy.ipynb
│   └── odornet_baseline_training.ipynb
├── src/
│   └── odornet/
├── README.md
├── data_dictionary.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── LICENSE-MIT
├── LICENSE-CC-BY-4.0
└── requirements.txt
```

## Citation

If you use OdorNet, please cite the dataset repository and the associated paper once available.

For the current release, use `CITATION.cff`. The dataset DOI is `10.5281/zenodo.19838456`; journal and final publication metadata are to be confirmed.

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
