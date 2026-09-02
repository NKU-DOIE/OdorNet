# Public Supporting Resources

This directory contains small, redistributable supporting resources used to
document the OdorNet taxonomy and provenance. It does not contain source-rich
raw molecule tables or row-level provenance records; those files remain part
of the Zenodo data-release workflow.

## Fragrantica Notes Vocabulary

`fragrantica_notes_2026-08-22.csv` contains 1,702 displayed
term-category records transcribed from a 22-page PDF capture of the
Fragrantica notes page dated August 22, 2026. The source page assigns some
terms to more than one displayed category. Consequently, global
case-insensitive deduplication yields 1,698 unique terms, provided in
`fragrantica_notes_2026-08-22_unique_terms.csv`. The unique-term file also
records each term's displayed-category count.

These files are a transparent reference vocabulary for expert-review and
taxonomy documentation. They are not OdorNet labels, do not modify the
released SEA mapping, and should not be interpreted as an endorsement by or
affiliation with Fragrantica. Users remain responsible for complying with the
source website's terms and for attributing Fragrantica when the vocabulary is
used.

## Source Registry

`source_registry.csv` is the machine-readable source catalog used to build the
Zenodo package. It records the `source_id`, source name, proposed source time,
raw-record count, unique molecule count, descriptor coverage, DOI when
available, and source URL. Blank DOI values are deliberate: a source may be a
website or book without an assigned DOI. The release builder uses these Source
ID values in every Zenodo provenance row.
