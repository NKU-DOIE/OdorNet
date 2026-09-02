"""Build a reviewable Zenodo candidate package with compact source provenance.

The script never edits the current Zenodo record. It creates a new local
package below `outputs/zenodo/` for review before a maintainer uploads a new
Zenodo version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERSION = "1.1.0"
RESOURCE_DIR = ROOT / "data" / "resources"
RAW_SOURCE_PATH = ROOT / "data" / "raw" / "merged_8892_cleaned_251230.pkl"

# The supplied raw snapshot uses these source labels. They are converted to
# the release Source ID values before any Zenodo file is written.
RAW_SOURCE_TO_SOURCE_ID = {
    "arctander_1960": "arctander",
    "aromadb": "aromadb",
    "flavordb": "flavordb",
    "flavornet": "flavornet",
    "goodscents": "TGSC",
    "ifra_2019": "IFRA",
    "leffingwell": "Leffingwell",
    "sharma_2021a": "SMILES_to_smell",
    "sharma_2021b": "OlfactionBase",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_registry() -> pd.DataFrame:
    path = RESOURCE_DIR / "source_registry.csv"
    registry = pd.read_csv(path, keep_default_na=False)
    if registry["source_id"].duplicated().any():
        raise ValueError("source_registry.csv contains duplicate source_id values.")
    required = {
        "source_id",
        "source_name",
        "proposed_time",
        "raw_record_count",
        "unique_molecule_count",
        "descriptor_coverage",
        "doi",
        "source_url",
    }
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"source_registry.csv is missing columns: {sorted(missing)}")
    return registry


def enrich_source_dataframe(source_df: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Map supplied source labels to Source ID values and retain SEA inputs."""
    registry_lookup = registry.set_index("source_id").to_dict(orient="index")
    result = source_df.copy(deep=True)
    enriched_rows = []
    for source_list in result["Source"]:
        enriched_sources = []
        for source_record in source_list:
            record = dict(source_record)
            supplied_source_id = record.get("Source", "")
            source_id = RAW_SOURCE_TO_SOURCE_ID.get(supplied_source_id, supplied_source_id)
            if source_id not in registry_lookup:
                raise KeyError(
                    f"Source registry has no entry for source {supplied_source_id!r}."
                )
            source_meta = registry_lookup[source_id]
            record.pop("Source", None)
            record["source_id"] = source_id
            record["doi"] = source_meta["doi"]
            record["source_url"] = source_meta["source_url"]
            enriched_sources.append(record)
        enriched_rows.append(enriched_sources)
    result["Source"] = enriched_rows
    return result


def flatten_source_records(source_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source_row in source_df.itertuples():
        for source_record in source_row.Source:
            rows.append(
                {
                    "SMILES": source_row.SMILES,
                    "source_id": source_record.get("source_id", ""),
                    "Original_SMILES": source_record.get("Original_SMILES", ""),
                    "Original_IUPACname": source_record.get("Original_IUPACname", ""),
                    "Original_Labels": source_record.get("Original_Labels", ""),
                    "doi": source_record.get("doi", ""),
                    "source_url": source_record.get("source_url", ""),
                }
            )
    return pd.DataFrame(rows)


def compact_source_lists(source_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only source identity and original evidence in nested release records."""
    result = source_df.copy(deep=True)
    compact_rows = []
    for source_list in result["Source"]:
        compact_rows.append(
            [
                {
                    "source_id": source_record.get("source_id", ""),
                    "Original_SMILES": source_record.get("Original_SMILES", ""),
                    "Original_IUPACname": source_record.get("Original_IUPACname", ""),
                    "Original_Labels": source_record.get("Original_Labels", ""),
                    "doi": source_record.get("doi", ""),
                    "source_url": source_record.get("source_url", ""),
                }
                for source_record in source_list
            ]
        )
    result["Source"] = compact_rows
    return result


def write_policy_tables(full: pd.DataFrame, processed_dir: Path) -> None:
    """Write explicit base, Drop, Union, and Intersection target tables."""
    label_columns = [
        column
        for column in full.columns
        if column not in {"SMILES", "Source"}
    ]
    base = full.copy()
    base.insert(1, "integration_policy", "unresolved_base")
    base.to_csv(processed_dir / "full_dataset.csv", index=False, na_rep="")

    drop = full.copy()
    drop.insert(1, "integration_policy", "Drop")
    drop.to_csv(processed_dir / "full_dataset_drop.csv", index=False, na_rep="")

    union = full.copy()
    union[label_columns] = union[label_columns].fillna(1).astype(int)
    union.insert(1, "integration_policy", "Union")
    union.to_csv(processed_dir / "full_dataset_union.csv", index=False)

    intersection = full.copy()
    intersection[label_columns] = intersection[label_columns].fillna(0).astype(int)
    intersection.insert(1, "integration_policy", "Intersection")
    intersection.to_csv(processed_dir / "full_dataset_intersection.csv", index=False)


def write_checksums(package_dir: Path) -> None:
    lines = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name == "checksums_sha256.txt":
            continue
        relative = path.relative_to(package_dir).as_posix()
        lines.append(f"{sha256_file(path)}  {relative}")
    (package_dir / "checksums_sha256.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_archive(package_dir: Path, output_root: Path) -> Path:
    archive_path = output_root / f"{package_dir.name}.zip"
    if archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {archive_path}")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_root))
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{sha256_file(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    return archive_path


def archive_readme(version: str) -> str:
    return f"""# OdorNet {version} Zenodo Candidate

This local package is a review candidate for a new Zenodo version. It does not
modify the already published Zenodo record.

## Provenance Clarifications

`data/provenance/source_records.csv` and
`data/provenance/source_records.jsonl` are compact row-wise provenance tables.
Each row contains the standardized `SMILES`, the `source_id`, the original
source SMILES and name when supplied, the original source labels, and the
source DOI and web link. Source-level dates and coverage statistics are kept in
`docs/source_catalog.md` and `data/provenance/source_registry.csv`.

## Aggregation Modes

`data/processed/full_dataset.csv` is the unresolved SEA base table. The
package additionally includes explicit `full_dataset_drop.csv`,
`full_dataset_union.csv`, and `full_dataset_intersection.csv` files. The
`integration_policy` column identifies their construction. Their states are:

- **Drop**: retain blanks as missing values and mask them from loss and metrics.
- **Union**: convert blanks to `1` after loading.
- **Intersection**: convert blanks to `0` after loading.

The raw source records and the aligned source-rich `full_dataset.csv` are
shared inputs for all three modes. The three modes are derived targets; they
are not different raw-data tables.

## Package Contents

- `data/processed/`: base/policy-specific full tables and released
  train/validation/test tables with the split manifest.
- `data/provenance/`: compact row-wise source records and source catalog.
- `data/metadata/`: SEA mappings and English descriptor definitions.
- `data/raw/`: enriched source-level molecule table used to reproduce labels.
- `data_dictionary.md`: GitHub-side field definitions and usage notes.
- `docs/source_catalog.md`: source-level names, proposed times, counts,
  descriptor coverage, DOI values, and web links.
- `RELEASE_NOTES.md`: release-specific changes and compatibility notes.
- `checksums_sha256.txt`: SHA-256 checksums for every package file.
"""


def build_package(version: str, output_root: Path) -> tuple[Path, Path]:
    package_dir = output_root / f"OdorNet_v{version}"
    if package_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing package directory: {package_dir}")
    output_root.mkdir(parents=True, exist_ok=True)
    package_dir.mkdir(parents=True)

    registry = _source_registry()
    source_df = pd.read_pickle(RAW_SOURCE_PATH)
    enriched_source_df = enrich_source_dataframe(source_df, registry)
    public_source_df = compact_source_lists(enriched_source_df)
    provenance_df = flatten_source_records(public_source_df)

    processed_dir = package_dir / "data" / "processed"
    provenance_dir = package_dir / "data" / "provenance"
    metadata_dir = package_dir / "data" / "metadata"
    raw_dir = package_dir / "data" / "raw"
    for directory in (processed_dir, provenance_dir, metadata_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(ROOT / "data" / "processed" / "full_dataset.csv")
    source_lookup = public_source_df.set_index("SMILES")["Source"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    if source_lookup.isna().any():
        raise ValueError("Could not attach enriched source records to every full-dataset row.")
    full.insert(1, "Source", full["SMILES"].map(source_lookup))
    write_policy_tables(full, processed_dir)
    shutil.copy2(ROOT / "data" / "processed" / "dataset_train_aligned.csv", processed_dir)
    shutil.copy2(ROOT / "data" / "processed" / "dataset_val_aligned.csv", processed_dir)
    shutil.copy2(ROOT / "data" / "processed" / "dataset_test_aligned.csv", processed_dir)
    shutil.copy2(ROOT / "data" / "processed" / "split_manifest.json", processed_dir)

    provenance_df.to_csv(provenance_dir / "source_records.csv", index=False)
    provenance_df.to_json(
        provenance_dir / "source_records.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
    registry.to_csv(provenance_dir / "source_registry.csv", index=False)
    enriched_source_df.to_pickle(raw_dir / "merged_8892_cleaned_251230.pkl")

    for name in (
        "olfactory_classification_strong_weak.json",
        "final_specialist_label_mapping.json",
        "olfactory_data_translated.json",
    ):
        shutil.copy2(ROOT / "data" / "metadata" / name, metadata_dir / name)
    shutil.copy2(ROOT / "data_dictionary.md", package_dir / "data_dictionary.md")
    docs_dir = package_dir / "docs"
    docs_dir.mkdir()
    shutil.copy2(ROOT / "docs" / "source_catalog.md", docs_dir / "source_catalog.md")
    (package_dir / "RELEASE_NOTES.md").write_text(
        "# OdorNet 1.1.0 Release Notes\n\n"
        "- Replaced the legacy two-way release split with the published 7:2:1 "
        "train/validation/test split (6,224/1,778/890 rows).\n"
        "- Enforced connectivity-group separation and verified no missing labels "
        "in validation or test.\n"
        "- Added explicit Drop, Union, and Intersection full-dataset targets.\n"
        "- Simplified source provenance to Source ID, original molecule and label "
        "fields, DOI, and web links; source statistics remain in the catalog.\n"
        "- Corrected the primary label spelling to `pungent&disagreeable`.\n",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text(archive_readme(version), encoding="utf-8")
    write_checksums(package_dir)
    archive_path = write_archive(package_dir, output_root)
    return package_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "zenodo")
    args = parser.parse_args()
    package_dir, archive_path = build_package(args.version, args.output_root)
    print(f"Package directory: {package_dir}")
    print(f"Archive: {archive_path}")
    print(f"Archive SHA-256: {sha256_file(archive_path)}")


if __name__ == "__main__":
    main()
