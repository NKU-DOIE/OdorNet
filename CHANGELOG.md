# Changelog

## v1.1.0 release preparation

- Added public Fragrantica reference-vocabulary resources with 1,702 displayed
  term-category records and 1,698 globally deduplicated terms.
- Replaced the legacy two-way train/test release with the published
  connectivity-safe 7:2:1 train/validation/test split (6,224/1,778/890 rows).
  Validation and test contain no unresolved label cells.
- Added `split_manifest.json` and a unified `technical_validation.ipynb` with
  source preprocessing, SEA, label-space, split, comparison, and visualization
  stages.
- Added a canonical nine-source registry, source catalog, provenance fields,
  and a Zenodo package builder with explicit Drop, Union, and Intersection
  full-dataset tables.
- Corrected the prior misspelling in the public label interface and
  standardized `SMILES` spelling.
- Added fixed-threshold raw multi-dataset and SEA integration-mode results to
  the technical-validation documentation.

## v1.0.0 manuscript release candidate - 2026-04-27

- Initial public release candidate for the OdorNet aligned molecule-level dataset.
- Added SEA taxonomy documentation for repository users.
- Documented curated missing-label policies: Union, Intersection, and Drop.
- Included aligned full, train, and test CSV files for 8,892 unique molecules.
- Organized dataset files under `data/`, notebooks under `notebooks/`, and runnable examples under `examples/`.
- Split licensing into `LICENSE`, `LICENSE-MIT`, and `LICENSE-CC-BY-4.0`.
- Kept the GitHub data release focused on machine-learning-ready processed CSV files.
- Moved source-rich provenance, raw curation data, and taxonomy metadata to the Zenodo archive workflow.
- Added Source JSON parsing in the public loader and Zenodo archive download/checksum verification helper.
- Added reusable processing code under `src/odornet/`.
- Added English-only SEA processing, split-strategy, and baseline training notebooks under `notebooks/`.
- Added repository metadata files for citation, licensing, requirements, and data loading.
