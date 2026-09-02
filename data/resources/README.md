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

`source_registry.csv` is the canonical source-level registry used to build the
Zenodo package. It records the stable manuscript-facing `source_id`, the
supplied `legacy_source_id`, source type, bibliographic reference, DOI or
source URL, time range, raw-record count, unique molecule count, descriptor
coverage, annotation procedure, and documented source relationships. Blank
DOI values are deliberate: a source may be a website or book without an
assigned DOI. The release builder joins nested source records to this registry
through `legacy_source_id`, then writes the canonical `source_id` to every
Zenodo provenance row.
