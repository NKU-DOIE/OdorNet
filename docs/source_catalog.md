# OdorNet Source Catalog

This catalog is the source-level companion to `data/resources/source_registry.csv`.
The canonical `source_id` values below are the labels used in the manuscript and
in the next Zenodo package. `legacy_source_id` records the pre-release labels
used in the working source table so that the conversion remains auditable.

The molecule and label counts reproduce Table 1 supplied with the technical
validation materials. Counts are unique molecules and unique source labels
within each constituent source; they are not row counts after consolidation.

| Canonical source ID | Legacy ID | Source | Time range represented | Raw records | Unique molecules | Descriptor coverage | DOI | Web link |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `OlfactionBase` | `sharma_2021b` | OlfactionBase | 2021-02-22 to source snapshot 2025-12-30 | 5,103 | 5,098 | 501 | `10.1021/acs.jcim.0c01288` | https://olfactionbase.com |
| `TGSC` | `goodscents` | The Good Scents Company | 1980-2025 website copyright; source snapshot 2025-12-30 | 4,622 | 4,561 | 667 | n/a | https://www.thegoodscentscompany.com |
| `SMILES_to_smell` | `sharma_2021a` | SMILES to Smell | 2021-02-22 | 4,006 | 4,006 | 572 | `10.1021/acs.jcim.0c01288` | https://figshare.com/articles/journal_contribution/SMILES_to_Smell_Decoding_the_Structure_Odor_Relationship_of_Chemical_Compounds_Using_the_Deep_Neural_Network_Approach/13584946 |
| `Leffingwell` | `leffingwell` | Leffingwell Odor Database | undated website; source snapshot 2025-12-30 | 3,522 | 3,522 | 113 | n/a | https://www.leffingwell.com/odordata.htm |
| `arctander` | `arctander_1960` | Arctander's dataset | 1960 | 2,824 | 2,751 | 77 | no assigned DOI | https://catalog.hathitrust.org/Record/001519716 |
| `IFRA` | `ifra_2019` | International Fragrance Association | 2019; source snapshot 2025-12-30 | 1,060 | 1,060 | 181 | n/a | https://ifrafragrance.org |
| `aromadb` | `aromadb` | AromaDB | 2018-08-13 | 869 | 869 | 127 | `10.3389/fpls.2018.01081` | https://aromadb.org |
| `flavornet` | `flavornet` | Flavornet | 1998 | 716 | 716 | 195 | `10.1016/S0167-4501(98)80029-0` | https://www.flavornet.org |
| `flavordb` | `flavordb` | FlavorDB | 2018-01-04 | 525 | 525 | 255 | `10.1093/nar/gkx957` | https://cosylab.iiitd.edu.in/flavordb |
| **OdorNet** | n/a | Consolidated dataset | release snapshot 2025-12-30 | **23,247*** | **8,892** | **556** | n/a | https://github.com/NKU-DOIE/OdorNet |

*The source-record count is 23,247 before structure consolidation. The RDKit
audit reports 23,247 parsed records, zero parsing failures, zero sanitization
removals, and 8,892 unique standardized SMILES.

## Provenance Fields

The Zenodo `data/provenance/source_records.csv` table contains one row for
each source-level record and preserves both the raw and normalized annotation
fields. The principal fields are:

- `source_id`: canonical source label used in the manuscript and release.
- `legacy_source_id`: source label in the pre-release working table.
- `source_record_id`: deterministic row identifier assigned during release
  packaging.
- `original_identifier` and `original_identifier_type`: source-provided SMILES
  and its explicit type, because no source-native record identifier is present
  in the consolidated snapshot.
- `Original_SMILES`: source-provided structure string.
- `Original_Labels`: raw source label text.
- `Pre_Processed_Labels`: parsed descriptor list before final normalization.
- `Processed_Labels`: normalized descriptor list used as SEA input.
- `raw_label` and `processed_label`: explicit aliases of the two label fields
  above for downstream tools.
- `source_doi`, `source_doi_status`, and `source_url`: persistent identifier
  and web provenance from the source registry.
- `time_range`: publication year or source-snapshot range used for this release.
- `raw_record_count`, `unique_molecule_count`, and `descriptor_coverage`:
  source-level summary values used in the manuscript table.
- `annotation_procedure` and `source_relationships`: how labels were captured
  and any known supplied-source relationship or aggregation note.

The source records may include repeated molecules because one molecule can have
multiple source annotations. The consolidated table contains 8,892 unique
isomeric SMILES, while the source-level table preserves the evidence needed to
inspect how those labels were combined.

## Source Relationships

`SMILES_to_smell` and `OlfactionBase` share the 2021 Sharma et al. DOI because
the released source materials are associated with the same publication. The
OlfactionBase entry additionally identifies the odorless/PubChem compilation
used for the relevant records. The other sources are independent public books,
web databases, or literature datasets; overlapping molecules are retained at
the source-record level and resolved at the consolidated label level by SEA.
