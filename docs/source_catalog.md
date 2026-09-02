# OdorNet Source Catalog

This catalog lists the nine original sources integrated into OdorNet. `Source
ID` is the identifier used throughout the release. The proposed time is the
publication or initial source date; no later snapshot range is included here.

| Source ID | Source | Proposed time | Raw records | Unique molecules | Descriptor coverage | DOI | Web link |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| `OlfactionBase` | OlfactionBase | 2021-02-22 | 5,103 | 5,098 | 501 | `10.1021/acs.jcim.0c01288` | https://olfactionbase.com |
| `TGSC` | The Good Scents Company | 1980 | 4,622 | 4,561 | 667 |  | https://www.thegoodscentscompany.com |
| `SMILES_to_smell` | SMILES to Smell | 2021-02-22 | 4,006 | 4,006 | 572 | `10.1021/acs.jcim.0c01288` | https://figshare.com/articles/journal_contribution/SMILES_to_Smell_Decoding_the_Structure_Odor_Relationship_of_Chemical_Compounds_Using_the_Deep_Neural_Network_Approach/13584946 |
| `Leffingwell` | Leffingwell Odor Database | undated | 3,522 | 3,522 | 113 |  | https://www.leffingwell.com/odordata.htm |
| `arctander` | Arctander's dataset | 1960 | 2,824 | 2,751 | 77 |  | https://catalog.hathitrust.org/Record/001519716 |
| `IFRA` | International Fragrance Association | 2019 | 1,060 | 1,060 | 181 |  | https://ifrafragrance.org |
| `aromadb` | AromaDB | 2018-08-13 | 869 | 869 | 127 | `10.3389/fpls.2018.01081` | https://aromadb.org |
| `flavornet` | Flavornet | 1998 | 716 | 716 | 195 | `10.1016/S0167-4501(98)80029-0` | https://www.flavornet.org |
| `flavordb` | FlavorDB | 2018-01-04 | 525 | 525 | 255 | `10.1093/nar/gkx957` | https://cosylab.iiitd.edu.in/flavordb |

## Consolidated Dataset

The release contains 23,247 source-level records and 8,892 unique standardized
SMILES. Descriptor coverage is reported per source above. A molecule may occur
in more than one source, and the source-level table preserves those separate
records.

## Source Records

`data/provenance/source_records.csv` and `data/provenance/source_records.jsonl`
contain one row per source annotation. They retain the standardized molecule
SMILES, `source_id`, original source SMILES and name when supplied, original
source labels, DOI, and source web link. Source dates and source-level counts
are kept in this catalog rather than repeated on every row.

The raw source snapshot used to rebuild SEA retains `Processed_Labels` in
`data/raw/merged_8892_cleaned_251230.pkl`; this internal field is not repeated
in the compact row-wise provenance tables.

## Source Relationships

`SMILES_to_smell` and `OlfactionBase` use the same Sharma et al. DOI because the
released materials are associated with the same publication. OlfactionBase
also contains the odorless/PubChem compilation used for the relevant records,
including the supplied beryllium-related entries. This statement describes
data provenance only and does not make an independent toxicity claim.
