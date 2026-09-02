"""Reproducible evaluations added for the OdorNet reviewer revision.

The helpers in this module operate on the released SEA mapping and on local
Zenodo-restored source files. They intentionally separate:

* taxonomy-level audits (semantic consistency and descriptor coverage);
* input-label deletion stability conditional on the released taxonomy; and
* preparation of GS_lf for the same Double-Drop target construction.

The original external AI voting artifacts are not part of the public release.
Accordingly, the deletion stability analysis re-applies the released mapping
and Double-Drop aggregation after perturbing source annotations; it does not
claim to re-run external AI semantic voting.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from ast import literal_eval
from collections.abc import Iterable
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import LABEL_COLUMNS
from .sea import (
    DEFAULT_MAPPING_PATH,
    DEFAULT_SEMANTIC_DESCRIPTION_PATH,
    DEFAULT_SPECIALIST_MAPPING_PATH,
    attach_main_labels,
    build_double_drop_matrix,
    build_reverse_mapping,
    load_main_label_mapping,
    load_semantic_descriptions,
    split_perfect_stratified,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NOT_ANY_TYPE = "not any type"
DEFAULT_GS_LF_PATH = REPO_ROOT / "data" / "raw" / "gs_lf.csv"
DEFAULT_SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"


def _read_json(path: Path | str) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_term(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _parse_label_list(value: object) -> list[str]:
    """Parse GS_lf's serialized Python label list without evaluating code."""
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        try:
            parsed = literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = [part for part in value.replace("|", ";").replace(",", ";").split(";")]
        values = parsed if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        values = []
    return sorted({_normalized_term(item) for item in values if str(item).strip()})


def build_secondary_assignment_table(
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    specialist_mapping_path: Path | str = DEFAULT_SPECIALIST_MAPPING_PATH,
) -> pd.DataFrame:
    """Return one row per SEA secondary-descriptor assignment.

    Strong, weak, expert-correction, and excluded assignments are retained as
    separate evidence rows. `odorless` is added explicitly because it is a
    released primary label but is not stored in the original mapping JSON.
    """
    statistical_mapping = _read_json(mapping_path)
    specialist_mapping = _read_json(specialist_mapping_path)
    rows: list[dict[str, str]] = []

    for category, descriptors in statistical_mapping.items():
        if category == NOT_ANY_TYPE:
            primary_category = ""
            tier = "excluded"
        elif category.endswith("_weak"):
            primary_category = category.removesuffix("_weak")
            tier = "weak"
        else:
            primary_category = category
            tier = "strong"
        for descriptor in descriptors:
            rows.append(
                {
                    "descriptor": _normalized_term(descriptor),
                    "primary_category": primary_category,
                    "assignment_tier": tier,
                    "assignment_source": "statistical_ai_mapping",
                }
            )

    for category, descriptors in specialist_mapping.items():
        for descriptor in descriptors:
            rows.append(
                {
                    "descriptor": _normalized_term(descriptor),
                    "primary_category": category,
                    "assignment_tier": "expert",
                    "assignment_source": "specialist_mapping",
                }
            )

    rows.append(
        {
            "descriptor": "odorless",
            "primary_category": "odorless",
            "assignment_tier": "strong",
            "assignment_source": "released_label_definition",
        }
    )

    return (
        pd.DataFrame(rows)
        .drop_duplicates()
        .sort_values(["descriptor", "primary_category", "assignment_tier"])
        .reset_index(drop=True)
    )


def _descriptor_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for descriptor, group in assignments.groupby("descriptor", sort=True):
        primary_categories = sorted(
            {
                category
                for category in group["primary_category"]
                if category in LABEL_COLUMNS
            }
        )
        tiers = sorted(set(group["assignment_tier"]))
        rows.append(
            {
                "descriptor": descriptor,
                "primary_categories": "; ".join(primary_categories),
                "primary_category_count": len(primary_categories),
                "assignment_tiers": "; ".join(tiers),
                "has_excluded_assignment": bool((group["assignment_tier"] == "excluded").any()),
                "has_strong_assignment": bool((group["assignment_tier"] == "strong").any()),
                "has_weak_assignment": bool((group["assignment_tier"] == "weak").any()),
                "has_expert_assignment": bool((group["assignment_tier"] == "expert").any()),
            }
        )
    return pd.DataFrame(rows)


def _resolve_sentence_transformer(model_name: str) -> tuple[str, bool, str]:
    """Prefer a complete local Hugging Face snapshot when it is available."""
    requested_path = Path(model_name)
    if requested_path.exists():
        return str(requested_path), True, requested_path.name

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        )
    )
    repository = "models--" + model_name.replace("/", "--")
    snapshot_root = cache_root / repository / "snapshots"
    snapshots = sorted(
        (
            path
            for path in snapshot_root.glob("*")
            if (path / "modules.json").is_file()
            and ((path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file())
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if snapshots:
        return str(snapshots[0]), True, snapshots[0].name
    return model_name, False, "remote_or_standard_cache"


def evaluate_semantic_consistency(
    assignments: pd.DataFrame,
    descriptions: dict[str, dict[str, object]] | None = None,
    model_name: str = DEFAULT_SENTENCE_TRANSFORMER,
    max_seq_length: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Calculate within- and cross-primary cosine similarity of definitions.

    Sentence embeddings are calculated once per unique descriptor. Cross-primary
    pairs containing the identical descriptor are omitted, so a descriptor that
    is legitimately assigned to multiple primary categories does not add an
    artificial cosine similarity of 1.0.
    """
    descriptions = descriptions or load_semantic_descriptions(DEFAULT_SEMANTIC_DESCRIPTION_PATH)
    descriptor_summary = _descriptor_summary(assignments)
    descriptor_summary = descriptor_summary[descriptor_summary["primary_category_count"] > 0].copy()
    normalized_descriptions = {
        _normalized_term(descriptor): details
        for descriptor, details in descriptions.items()
    }
    descriptor_summary["english_definition"] = descriptor_summary["descriptor"].map(
        lambda descriptor: str(
            normalized_descriptions.get(descriptor, {}).get("content", "")
        ).strip()
    )
    descriptor_summary["has_english_definition"] = descriptor_summary["english_definition"].ne("")
    encoded = descriptor_summary[descriptor_summary["has_english_definition"]].copy()
    if encoded.empty:
        raise ValueError("No English secondary-descriptor definitions are available for semantic evaluation.")

    from sentence_transformers import SentenceTransformer

    resolved_model, local_files_only, model_revision = _resolve_sentence_transformer(model_name)
    model = SentenceTransformer(resolved_model, local_files_only=local_files_only)
    model.max_seq_length = max_seq_length
    texts = [
        f"Odor descriptor: {row.descriptor}. Definition: {row.english_definition}"
        for row in encoded.itertuples()
    ]
    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    descriptor_to_index = {descriptor: idx for idx, descriptor in enumerate(encoded["descriptor"])}
    descriptor_to_categories = {
        row.descriptor: [item for item in row.primary_categories.split("; ") if item]
        for row in encoded.itertuples()
    }
    primary_to_descriptors = {
        primary: sorted(
            descriptor
            for descriptor, categories in descriptor_to_categories.items()
            if primary in categories
        )
        for primary in LABEL_COLUMNS
    }

    within_rows = []
    for primary, descriptors in primary_to_descriptors.items():
        indices = [descriptor_to_index[item] for item in descriptors]
        if len(indices) < 2:
            mean_similarity = np.nan
            pair_count = 0
        else:
            submatrix = embeddings[indices] @ embeddings[indices].T
            upper = submatrix[np.triu_indices(len(indices), k=1)]
            mean_similarity = float(np.mean(upper))
            pair_count = int(len(upper))
        within_rows.append(
            {
                "primary_category": primary,
                "secondary_descriptor_count": len(descriptors),
                "definition_count": len(descriptors),
                "pair_count": pair_count,
                "mean_cosine_similarity": mean_similarity,
            }
        )
    within_df = pd.DataFrame(within_rows)

    cross_rows = []
    for left_index, left_primary in enumerate(LABEL_COLUMNS):
        for right_primary in LABEL_COLUMNS[left_index + 1 :]:
            left_descriptors = primary_to_descriptors[left_primary]
            right_descriptors = primary_to_descriptors[right_primary]
            similarities = []
            shared = 0
            for left_descriptor in left_descriptors:
                for right_descriptor in right_descriptors:
                    if left_descriptor == right_descriptor:
                        shared += 1
                        continue
                    similarities.append(
                        float(
                            embeddings[descriptor_to_index[left_descriptor]]
                            @ embeddings[descriptor_to_index[right_descriptor]]
                        )
                    )
            cross_rows.append(
                {
                    "primary_category_left": left_primary,
                    "primary_category_right": right_primary,
                    "secondary_descriptor_count_left": len(left_descriptors),
                    "secondary_descriptor_count_right": len(right_descriptors),
                    "shared_descriptor_count_excluded": shared,
                    "pair_count": len(similarities),
                    "mean_cosine_similarity": float(np.mean(similarities))
                    if similarities
                    else np.nan,
                }
            )
    cross_df = pd.DataFrame(cross_rows)

    summary = {
        "model_name": model_name,
        "model_revision": model_revision,
        "model_loaded_from_local_snapshot": local_files_only,
        "max_seq_length": max_seq_length,
        "descriptor_count_with_definition": int(len(encoded)),
        "descriptor_count_without_definition": int((~descriptor_summary["has_english_definition"]).sum()),
        "within_primary_mean_similarity_macro": float(within_df["mean_cosine_similarity"].mean()),
        "within_primary_mean_similarity_weighted": float(
            np.average(
                within_df["mean_cosine_similarity"].dropna(),
                weights=within_df.loc[within_df["mean_cosine_similarity"].notna(), "pair_count"],
            )
        ),
        "cross_primary_mean_similarity_macro": float(cross_df["mean_cosine_similarity"].mean()),
        "cross_primary_mean_similarity_weighted": float(
            np.average(
                cross_df["mean_cosine_similarity"].dropna(),
                weights=cross_df.loc[cross_df["mean_cosine_similarity"].notna(), "pair_count"],
            )
        ),
    }
    return descriptor_summary, within_df, cross_df, summary


def evaluate_label_coverage(
    source_df: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize source-vocabulary coverage by the released SEA mapping.

    The output reports assignment coverage, not externally validated
    correctness. A taxonomy-level ground truth for each raw descriptor is not
    included in the release and therefore cannot support a claim of accuracy.
    """
    molecule_level_counts: Counter[str] = Counter()
    source_record_counts: Counter[str] = Counter()
    molecule_sets: dict[str, set[str]] = {}
    source_record_molecule_sets: dict[str, set[str]] = {}
    source_ids: dict[str, set[str]] = {}
    for row in source_df.itertuples():
        for descriptor in row.Processed_Labels:
            normalized = _normalized_term(descriptor)
            molecule_level_counts[normalized] += 1
            molecule_sets.setdefault(normalized, set()).add(row.SMILES)
        for source_record in row.Source:
            source_id = str(source_record.get("Source", "")).strip()
            for descriptor in source_record.get("Processed_Labels", []):
                normalized = _normalized_term(descriptor)
                source_record_counts[normalized] += 1
                source_record_molecule_sets.setdefault(normalized, set()).add(row.SMILES)
                source_ids.setdefault(normalized, set()).add(source_id)

    assignment_summary = _descriptor_summary(assignments).set_index("descriptor")
    rows = []

    def append_scope_rows(scope: str, descriptors: set[str]) -> None:
        for descriptor in sorted(descriptors):
            info = assignment_summary.loc[descriptor] if descriptor in assignment_summary.index else None
            category_count = int(info["primary_category_count"]) if info is not None else 0
            excluded = bool(info["has_excluded_assignment"]) if info is not None else False
            if category_count == 1:
                status = "assigned_one_primary"
            elif category_count > 1:
                status = "assigned_multiple_primaries"
            elif excluded:
                status = "discarded_not_any_type"
            else:
                status = "unmapped"
            rows.append(
                {
                    "coverage_scope": scope,
                    "descriptor": descriptor,
                    "molecule_level_occurrences": molecule_level_counts[descriptor],
                    "molecule_level_molecule_count": len(molecule_sets.get(descriptor, set())),
                    "source_record_label_occurrences": source_record_counts[descriptor],
                    "source_record_molecule_count": len(
                        source_record_molecule_sets.get(descriptor, set())
                    ),
                    "source_ids": "; ".join(sorted(source_ids.get(descriptor, set()))),
                    "primary_categories": info["primary_categories"] if info is not None else "",
                    "primary_category_count": category_count,
                    "assignment_tiers": info["assignment_tiers"] if info is not None else "",
                    "has_excluded_assignment": excluded,
                    "coverage_status": status,
                }
            )

    append_scope_rows("molecule_level", set(molecule_level_counts))
    append_scope_rows("source_record", set(source_record_counts))
    detail_df = pd.DataFrame(rows)
    summary_df = (
        detail_df.groupby(["coverage_scope", "coverage_status"], sort=False)
        .agg(
            descriptor_count=("descriptor", "size"),
            molecule_level_occurrences=("molecule_level_occurrences", "sum"),
            source_record_label_occurrences=("source_record_label_occurrences", "sum"),
            molecule_level_molecule_count_sum=("molecule_level_molecule_count", "sum"),
            source_record_molecule_count_sum=("source_record_molecule_count", "sum"),
        )
        .reset_index()
    )
    total_rows = []
    for scope, scope_df in detail_df.groupby("coverage_scope", sort=False):
        total_rows.append(
            {
                "coverage_scope": scope,
                "coverage_status": "total_source_descriptors",
                "descriptor_count": int(len(scope_df)),
                "molecule_level_occurrences": int(scope_df["molecule_level_occurrences"].sum()),
                "source_record_label_occurrences": int(
                    scope_df["source_record_label_occurrences"].sum()
                ),
                "molecule_level_molecule_count_sum": int(
                    scope_df["molecule_level_molecule_count"].sum()
                ),
                "source_record_molecule_count_sum": int(
                    scope_df["source_record_molecule_count"].sum()
                ),
            }
        )
    total = pd.DataFrame(total_rows)
    return detail_df, pd.concat([total, summary_df], ignore_index=True)


def _drop_exact_fraction_of_source_labels(
    source_df: pd.DataFrame,
    drop_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, int, int]:
    if not 0 < drop_fraction < 1:
        raise ValueError("drop_fraction must be between 0 and 1.")
    locations: list[tuple[int, int, int]] = []
    for row_index, source_list in enumerate(source_df["Source"]):
        for source_index, record in enumerate(source_list):
            for label_index, _ in enumerate(record.get("Processed_Labels", [])):
                locations.append((row_index, source_index, label_index))
    drop_count = int(round(len(locations) * drop_fraction))
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(len(locations), size=drop_count, replace=False).tolist())
    drop_locations = {locations[index] for index in chosen}

    result = source_df.copy(deep=True)
    updated_rows = []
    for row_index, source_list in enumerate(result["Source"]):
        updated_sources = []
        for source_index, record in enumerate(copy.deepcopy(source_list)):
            labels = record.get("Processed_Labels", [])
            record["Processed_Labels"] = [
                label
                for label_index, label in enumerate(labels)
                if (row_index, source_index, label_index) not in drop_locations
            ]
            updated_sources.append(record)
        updated_rows.append(updated_sources)
    result["Source"] = updated_rows
    return result, len(locations), drop_count


def _matrix_agreement(
    baseline: pd.DataFrame,
    perturbed: pd.DataFrame,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    merged = baseline[["SMILES", *LABEL_COLUMNS]].merge(
        perturbed[["SMILES", *LABEL_COLUMNS]],
        on="SMILES",
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_perturbed"),
    )
    all_equal = []
    per_label = []
    all_positive_baseline = 0
    all_positive_perturbed = 0
    all_positive_intersection = 0
    for label in LABEL_COLUMNS:
        left = pd.to_numeric(merged[f"{label}_baseline"], errors="coerce").to_numpy()
        right = pd.to_numeric(merged[f"{label}_perturbed"], errors="coerce").to_numpy()
        equal = (left == right) | (np.isnan(left) & np.isnan(right))
        all_equal.append(equal)
        baseline_positive = left == 1.0
        perturbed_positive = right == 1.0
        intersection = baseline_positive & perturbed_positive
        union = baseline_positive | perturbed_positive
        all_positive_baseline += int(baseline_positive.sum())
        all_positive_perturbed += int(perturbed_positive.sum())
        all_positive_intersection += int(intersection.sum())
        per_label.append(
            {
                "label": label,
                "cell_agreement": float(equal.mean()),
                "cell_change_rate": float((~equal).mean()),
                "baseline_positive_count": int(baseline_positive.sum()),
                "perturbed_positive_count": int(perturbed_positive.sum()),
                "positive_jaccard": float(intersection.sum() / union.sum()) if union.any() else 1.0,
            }
        )
    all_equal_array = np.column_stack(all_equal)
    positive_union = all_positive_baseline + all_positive_perturbed - all_positive_intersection
    summary = {
        "label_cell_agreement": float(all_equal_array.mean()),
        "label_cell_change_rate": float((~all_equal_array).mean()),
        "molecule_exact_agreement": float(all_equal_array.all(axis=1).mean()),
        "baseline_positive_count": all_positive_baseline,
        "perturbed_positive_count": all_positive_perturbed,
        "positive_jaccard": float(all_positive_intersection / positive_union)
        if positive_union
        else 1.0,
    }
    return summary, pd.DataFrame(per_label)


def evaluate_label_deletion_stability(
    source_df: pd.DataFrame,
    mapping: dict[str, list[str]] | None = None,
    drop_fraction: float = 0.2,
    seeds: list[int] | tuple[int, ...] = tuple(range(1001, 1011)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild SEA/Double-Drop labels after deleting an exact label fraction."""
    mapping = mapping or load_main_label_mapping()
    reverse_mapping = build_reverse_mapping(mapping)
    baseline = build_double_drop_matrix(attach_main_labels(source_df, reverse_mapping))
    repetition_rows = []
    per_label_rows = []
    for repetition, seed in enumerate(seeds, start=1):
        perturbed_source, input_count, dropped_count = _drop_exact_fraction_of_source_labels(
            source_df,
            drop_fraction=drop_fraction,
            seed=seed,
        )
        perturbed = build_double_drop_matrix(attach_main_labels(perturbed_source, reverse_mapping))
        summary, per_label = _matrix_agreement(baseline, perturbed)
        repetition_rows.append(
            {
                "repetition": repetition,
                "seed": seed,
                "source_label_instances": input_count,
                "dropped_label_instances": dropped_count,
                "drop_fraction": dropped_count / input_count,
                **summary,
            }
        )
        per_label.insert(0, "seed", seed)
        per_label_rows.append(per_label)
    return pd.DataFrame(repetition_rows), pd.concat(per_label_rows, ignore_index=True)


def prepare_gs_lf_matrix(
    gs_lf_path: Path | str = DEFAULT_GS_LF_PATH,
    mapping: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map GS_lf raw descriptors through the released SEA/Double-Drop rule."""
    raw = pd.read_csv(gs_lf_path)
    required = {"IsomericSMILES", "Labels"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"GS_lf is missing required columns: {sorted(missing)}")
    if raw["IsomericSMILES"].duplicated().any():
        raise ValueError("GS_lf contains duplicate IsomericSMILES values.")

    source_rows = []
    for row in raw.itertuples(index=False):
        labels = _parse_label_list(row.Labels)
        source_rows.append(
            {
                "SMILES": row.IsomericSMILES,
                "Source": [
                    {
                        "Source": "gs_lf",
                        "Original_Labels": row.Labels,
                        "Pre_Processed_Labels": labels,
                        "Processed_Labels": labels,
                    }
                ],
            }
        )
    source_df = pd.DataFrame(source_rows)
    mapping = mapping or load_main_label_mapping()
    mapped = attach_main_labels(source_df, build_reverse_mapping(mapping))
    matrix = build_double_drop_matrix(mapped)
    return matrix, source_df


def split_gs_lf_matrix(
    matrix: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 959,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create the matched fixed GS_lf split using the released split helper."""
    return split_perfect_stratified(matrix, test_size=test_size, seed=seed)


def randomize_primary_secondary_mapping(
    mapping: dict[str, list[str]],
    seed: int,
    swaps_per_edge: int = 100,
    max_attempt_multiplier: int = 1_000,
) -> dict[str, list[str]]:
    """Randomize descriptor-category edges with degree-preserving edge swaps.

    Strong/excluded and weak assignment pools are shuffled independently. A
    valid two-edge swap preserves both the number of descriptors per category
    and the number of categories assigned to each descriptor, while avoiding
    duplicate descriptor-category relations.
    """
    rng = np.random.default_rng(seed)
    randomized = {category: [] for category in mapping}
    for is_weak in (False, True):
        categories = [
            category
            for category in mapping
            if category.endswith("_weak") == is_weak
        ]
        pairs = [
            (descriptor, category)
            for category in categories
            for descriptor in sorted(set(mapping[category]))
        ]
        edges = list(pairs)
        edge_set = set(edges)
        target_swaps = max(len(edges) * swaps_per_edge, 1)
        max_attempts = max(target_swaps * max_attempt_multiplier, 1)
        completed_swaps = 0
        attempts = 0
        while completed_swaps < target_swaps and attempts < max_attempts:
            attempts += 1
            left_index, right_index = rng.integers(0, len(edges), size=2)
            if left_index == right_index:
                continue
            left_descriptor, left_category = edges[left_index]
            right_descriptor, right_category = edges[right_index]
            if left_descriptor == right_descriptor or left_category == right_category:
                continue
            left_rewired = (left_descriptor, right_category)
            right_rewired = (right_descriptor, left_category)
            if left_rewired in edge_set or right_rewired in edge_set:
                continue
            edge_set.remove((left_descriptor, left_category))
            edge_set.remove((right_descriptor, right_category))
            edge_set.add(left_rewired)
            edge_set.add(right_rewired)
            edges[left_index] = left_rewired
            edges[right_index] = right_rewired
            completed_swaps += 1
        if completed_swaps < target_swaps:
            raise RuntimeError(
                "Unable to complete the requested degree-preserving edge swaps."
            )
        for descriptor, category in edges:
            randomized[category].append(descriptor)
    return {category: sorted(values) for category, values in randomized.items()}


def mapping_sha256(mapping: dict[str, list[str]]) -> str:
    """Return a stable compact identifier for a mapping configuration."""
    payload = json.dumps(mapping, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def clean_smiles_with_rdkit(value: object) -> dict[str, object]:
    """Parse, sanitize, and canonicalize one source SMILES string.

    Parsing and sanitization are intentionally reported as separate stages so
    that an invalid string and a chemically inconsistent RDKit molecule are
    not counted as the same failure. The returned canonical key keeps
    stereochemistry because the released molecule table is isomeric.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {
            "raw_smiles": "",
            "status": "empty",
            "failure_stage": "empty",
            "failure_detail": "blank_or_null_input",
            "cleaned_smiles": "",
        }

    raw_smiles = str(value).strip()
    if not raw_smiles:
        return {
            "raw_smiles": raw_smiles,
            "status": "empty",
            "failure_stage": "empty",
            "failure_detail": "blank_input",
            "cleaned_smiles": "",
        }

    try:
        molecule = Chem.MolFromSmiles(raw_smiles, sanitize=False)
    except Exception as exc:  # pragma: no cover - defensive RDKit boundary
        return {
            "raw_smiles": raw_smiles,
            "status": "parse_failure",
            "failure_stage": "parse",
            "failure_detail": type(exc).__name__,
            "cleaned_smiles": "",
        }
    if molecule is None:
        return {
            "raw_smiles": raw_smiles,
            "status": "parse_failure",
            "failure_stage": "parse",
            "failure_detail": "MolFromSmiles_returned_None",
            "cleaned_smiles": "",
        }

    try:
        sanitize_code = Chem.SanitizeMol(molecule, catchErrors=True)
    except Exception as exc:  # pragma: no cover - defensive RDKit boundary
        return {
            "raw_smiles": raw_smiles,
            "status": "sanitize_failure",
            "failure_stage": "sanitize",
            "failure_detail": type(exc).__name__,
            "cleaned_smiles": "",
        }
    if int(sanitize_code) != 0:
        failure_detail = str(sanitize_code).split(".")[-1]
        return {
            "raw_smiles": raw_smiles,
            "status": "sanitize_failure",
            "failure_stage": "sanitize",
            "failure_detail": failure_detail,
            "cleaned_smiles": "",
        }

    try:
        cleaned_smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception as exc:  # pragma: no cover - defensive RDKit boundary
        return {
            "raw_smiles": raw_smiles,
            "status": "conversion_failure",
            "failure_stage": "canonicalization",
            "failure_detail": type(exc).__name__,
            "cleaned_smiles": "",
        }

    return {
        "raw_smiles": raw_smiles,
        "status": "valid",
        "failure_stage": "",
        "failure_detail": "",
        "cleaned_smiles": cleaned_smiles,
    }


def audit_source_smiles_preprocessing(
    source_df: pd.DataFrame,
    source_column: str = "Source",
    smiles_column: str = "SMILES",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Audit source-record SMILES cleaning and return a deduplicated table.

    The source pickle has one molecule row and a nested list of source
    records. Every nested ``Original_SMILES`` is treated as an input record;
    no record is removed before RDKit validation. Deduplication is performed
    only after successful canonicalization and uses canonical isomeric SMILES
    as the key.
    """
    rows: list[dict[str, object]] = []
    for molecule_row_index, source_row in enumerate(source_df.itertuples(index=False)):
        source_records = getattr(source_row, source_column)
        if source_records is None:
            source_records = []
        for source_record_index, source_record in enumerate(source_records):
            raw_value = source_record.get("Original_SMILES", "")
            cleaned = clean_smiles_with_rdkit(raw_value)
            rows.append(
                {
                    "source_row_index": molecule_row_index,
                    "source_record_index": source_record_index,
                    "source_id": source_record.get("Source", ""),
                    "released_row_smiles": getattr(source_row, smiles_column),
                    **cleaned,
                }
            )

    audit_df = pd.DataFrame(rows)
    valid_df = audit_df[audit_df["status"].eq("valid")].copy()
    duplicate_counts = (
        valid_df.groupby("cleaned_smiles", as_index=False)
        .agg(
            source_record_count=("cleaned_smiles", "size"),
            source_id_count=("source_id", "nunique"),
        )
    )
    deduplicated_df = (
        valid_df.sort_values(["cleaned_smiles", "source_row_index", "source_record_index"])
        .drop_duplicates("cleaned_smiles", keep="first")
        .merge(duplicate_counts, on="cleaned_smiles", how="left", validate="one_to_one")
        .sort_values("cleaned_smiles")
        .reset_index(drop=True)
    )

    status_counts = audit_df["status"].value_counts().to_dict()
    raw_unique_count = int(audit_df.loc[audit_df["raw_smiles"].ne(""), "raw_smiles"].nunique())
    valid_count = int(len(valid_df))
    unique_count = int(deduplicated_df["cleaned_smiles"].nunique())
    top_level_keys = set()
    top_level_failures = 0
    for value in source_df[smiles_column]:
        top_level_result = clean_smiles_with_rdkit(value)
        if top_level_result["status"] == "valid":
            top_level_keys.add(str(top_level_result["cleaned_smiles"]))
        else:
            top_level_failures += 1
    cleaned_smiles = set(deduplicated_df["cleaned_smiles"])
    summary = {
        "input_molecule_rows": int(len(source_df)),
        "input_source_records": int(len(audit_df)),
        "raw_nonempty_smiles_unique": raw_unique_count,
        "empty_smiles_records": int(status_counts.get("empty", 0)),
        "rdkit_parse_failures": int(status_counts.get("parse_failure", 0)),
        "rdkit_sanitize_failures": int(status_counts.get("sanitize_failure", 0)),
        "rdkit_canonicalization_failures": int(status_counts.get("conversion_failure", 0)),
        "invalid_or_unusable_records_removed": int(
            len(audit_df) - valid_count
        ),
        "valid_source_records": valid_count,
        "unique_cleaned_isomeric_smiles": unique_count,
        "duplicate_source_records_removed": valid_count - unique_count,
        "top_level_molecule_smiles": int(len(top_level_keys)),
        "top_level_smiles_parse_or_clean_failures": int(top_level_failures),
        "cleaned_set_matches_top_level_set": cleaned_smiles == top_level_keys,
        "top_level_only_smiles_count": int(len(top_level_keys - cleaned_smiles)),
        "cleaned_only_smiles_count": int(len(cleaned_smiles - top_level_keys)),
    }
    return audit_df, deduplicated_df, summary


def canonical_structure_keys(smiles: object) -> dict[str, object]:
    """Return canonical isomeric and connectivity keys for split audits."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return {
            "canonical_isomeric_smiles": "",
            "connectivity_smiles": "",
            "has_explicit_stereochemistry": False,
            "parse_status": "failure",
        }
    has_chiral = any(
        atom.GetChiralTag().name != "CHI_UNSPECIFIED" for atom in molecule.GetAtoms()
    )
    has_bond_stereo = any(
        bond.GetStereo().name != "STEREONONE" for bond in molecule.GetBonds()
    )
    return {
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        ),
        "connectivity_smiles": Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=False
        ),
        "has_explicit_stereochemistry": bool(has_chiral or has_bond_stereo),
        "parse_status": "valid",
    }


def search_stereo_safe_split(
    full_df: pd.DataFrame,
    test_size: float = 0.2,
    seed_candidates: Iterable[int] = range(1000),
    target_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Find a validation split with complete connectivity groups only.

    The validation set is restricted to connectivity groups whose molecules are
    all fully labeled, which guarantees that validation contains no NaN label
    cells. Connectivity groups are kept intact, so no stereoisomer pair can be
    split across train and validation.

    Candidate splits are generated with multilabel stratification on the
    eligible complete connectivity groups. The selected split minimizes the
    mean absolute difference between the final train and validation label
    prevalence rates, using exact validation size as the first tie-breaker.
    """
    target_labels = target_labels or LABEL_COLUMNS
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    working = full_df.copy(deep=True)
    key_df = working["SMILES"].map(canonical_structure_keys).apply(pd.Series)
    working = pd.concat([working.reset_index(drop=True), key_df.reset_index(drop=True)], axis=1)

    group_rows: list[dict[str, object]] = []
    heterogeneous_complete_groups = 0
    heterogeneous_complete_molecules = 0
    for connectivity, group in working.groupby("connectivity_smiles", sort=True):
        complete_mask = group[target_labels].notna().all(axis=1)
        if not bool(complete_mask.all()):
            continue
        label_frame = group[target_labels].apply(pd.to_numeric, errors="coerce")
        unique_label_rows = int(label_frame.drop_duplicates().shape[0])
        if unique_label_rows > 1:
            heterogeneous_complete_groups += 1
            heterogeneous_complete_molecules += len(group)
        row: dict[str, object] = {
            "connectivity_smiles": connectivity,
            "group_size": int(len(group)),
            "unique_label_rows": unique_label_rows,
        }
        for label in target_labels:
            values = pd.to_numeric(group[label], errors="coerce")
            row[f"{label}_positive_count"] = int((values == 1).sum())
            row[label] = int((values == 1).any())
        group_rows.append(row)

    group_df = pd.DataFrame(group_rows)
    if group_df.empty:
        raise ValueError("No all-complete connectivity groups were available for the split search.")

    target_validation_rows = int(round(len(working) * test_size))
    eligible_molecules = int(group_df["group_size"].sum())
    target_ratio = target_validation_rows / eligible_molecules
    matrix = group_df[target_labels].to_numpy(dtype=int)
    dummy_x = np.zeros((len(group_df), 1), dtype=int)

    search_rows: list[dict[str, object]] = []
    for seed in seed_candidates:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=target_ratio,
            random_state=int(seed),
        )
        train_idx, validation_idx = next(splitter.split(dummy_x, matrix))
        validation_groups = group_df.iloc[validation_idx]
        validation_connectivities = set(validation_groups["connectivity_smiles"])

        validation_df = working[working["connectivity_smiles"].isin(validation_connectivities)].copy()
        train_df = working[~working["connectivity_smiles"].isin(validation_connectivities)].copy()

        train_rates = train_df[target_labels].apply(pd.to_numeric, errors="coerce").mean()
        validation_rates = validation_df[target_labels].apply(pd.to_numeric, errors="coerce").mean()
        abs_diff = (train_rates - validation_rates).abs()
        search_rows.append(
            {
                "seed": int(seed),
                "validation_rows": int(len(validation_df)),
                "train_rows": int(len(train_df)),
                "validation_group_count": int(len(validation_groups)),
                "validation_size_error": int(abs(len(validation_df) - target_validation_rows)),
                "mean_abs_rate_diff": float(abs_diff.mean()),
                "max_abs_rate_diff": float(abs_diff.max()),
                "mean_abs_rate_diff_excl_odorless": float(
                    abs_diff.drop(index="odorless").mean()
                    if "odorless" in abs_diff.index
                    else abs_diff.mean()
                ),
            }
        )

    search_df = pd.DataFrame(search_rows)
    exact_size_df = search_df[search_df["validation_rows"].eq(target_validation_rows)].copy()
    ranking_df = exact_size_df if not exact_size_df.empty else search_df
    best_row = (
        ranking_df.sort_values(
            [
                "mean_abs_rate_diff",
                "max_abs_rate_diff",
                "validation_size_error",
                "seed",
            ]
        )
        .iloc[0]
        .to_dict()
    )
    best_seed = int(best_row["seed"])

    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=target_ratio,
        random_state=best_seed,
    )
    train_idx, validation_idx = next(splitter.split(dummy_x, matrix))
    validation_groups = group_df.iloc[validation_idx]
    validation_connectivities = set(validation_groups["connectivity_smiles"])
    validation_df = working[working["connectivity_smiles"].isin(validation_connectivities)].copy()
    train_df = working[~working["connectivity_smiles"].isin(validation_connectivities)].copy()

    train_rates = train_df[target_labels].apply(pd.to_numeric, errors="coerce").mean()
    validation_rates = validation_df[target_labels].apply(pd.to_numeric, errors="coerce").mean()
    abs_diff = (train_rates - validation_rates).abs()
    metadata = {
        "strategy": "complete_group_multilabel_stratification",
        "test_size_requested": float(test_size),
        "test_size_applied_on_complete_groups": float(target_ratio),
        "target_validation_rows": int(target_validation_rows),
        "actual_validation_rows": int(len(validation_df)),
        "actual_train_rows": int(len(train_df)),
        "eligible_complete_group_count": int(len(group_df)),
        "eligible_complete_molecule_count": eligible_molecules,
        "validation_group_count": int(len(validation_groups)),
        "validation_group_fraction_of_eligible_groups": float(len(validation_groups) / len(group_df)),
        "validation_row_fraction_of_total": float(len(validation_df) / len(working)),
        "seed_search_count": int(len(search_df)),
        "exact_size_candidate_count": int(len(exact_size_df)),
        "selected_seed": best_seed,
        "mean_abs_rate_diff": float(abs_diff.mean()),
        "max_abs_rate_diff": float(abs_diff.max()),
        "mean_abs_rate_diff_excl_odorless": float(
            abs_diff.drop(index="odorless").mean()
            if "odorless" in abs_diff.index
            else abs_diff.mean()
        ),
        "validation_has_nan_cells": int(validation_df[target_labels].isna().sum().sum()),
        "train_has_nan_cells": int(train_df[target_labels].isna().sum().sum()),
        "shared_connectivity_count": int(
            len(set(train_df["connectivity_smiles"]) & set(validation_df["connectivity_smiles"]))
        ),
        "heterogeneous_complete_group_count": heterogeneous_complete_groups,
        "heterogeneous_complete_molecule_count": heterogeneous_complete_molecules,
        "label_rate_table": [
            {
                "label": label,
                "train_support": int(train_df[label].notna().sum()),
                "train_positive": int((pd.to_numeric(train_df[label], errors="coerce") == 1).sum()),
                "train_rate": float(train_rates[label]),
                "validation_support": int(validation_df[label].notna().sum()),
                "validation_positive": int((pd.to_numeric(validation_df[label], errors="coerce") == 1).sum()),
                "validation_rate": float(validation_rates[label]),
                "absolute_difference": float(abs_diff[label]),
            }
            for label in target_labels
        ],
    }
    return train_df, validation_df, search_df, metadata
