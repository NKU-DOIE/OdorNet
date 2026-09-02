"""Run reviewer-requested SEA audits and controlled model comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from odornet.datasets import LABEL_COLUMNS, load_odornet, load_source_metadata
from odornet.reviewer_evaluations import (
    build_secondary_assignment_table,
    evaluate_label_coverage,
    evaluate_label_deletion_stability,
    evaluate_semantic_consistency,
    mapping_sha256,
    prepare_gs_lf_matrix,
    randomize_primary_secondary_mapping,
    search_stereo_safe_split,
    split_gs_lf_matrix,
)
from odornet.sea import attach_main_labels, build_double_drop_matrix, build_reverse_mapping, load_main_label_mapping
from odornet.training import TrainingConfig, train_gnn_baseline, train_molformer_baseline


DEFAULT_RESULTS_DIR = ROOT / "results" / "reviewer_revision"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "reviewer_revision"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _plot_semantic_matrix(within_df: pd.DataFrame, cross_df: pd.DataFrame, path: Path) -> None:
    matrix = pd.DataFrame(np.nan, index=LABEL_COLUMNS, columns=LABEL_COLUMNS)
    for row in within_df.itertuples():
        matrix.loc[row.primary_category, row.primary_category] = row.mean_cosine_similarity
    for row in cross_df.itertuples():
        matrix.loc[row.primary_category_left, row.primary_category_right] = row.mean_cosine_similarity
        matrix.loc[row.primary_category_right, row.primary_category_left] = row.mean_cosine_similarity

    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(range(len(LABEL_COLUMNS)), LABEL_COLUMNS, rotation=55, ha="right")
    axis.set_yticks(range(len(LABEL_COLUMNS)), LABEL_COLUMNS)
    for row_index in range(len(LABEL_COLUMNS)):
        for column_index in range(len(LABEL_COLUMNS)):
            value = matrix.iat[row_index, column_index]
            if not np.isnan(value):
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, label="Mean cosine similarity")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_taxonomy_audits(results_dir: Path, deletion_repetitions: int) -> None:
    assignments = build_secondary_assignment_table()
    source_df = load_source_metadata(root=ROOT)
    descriptor_detail, coverage_summary = evaluate_label_coverage(source_df, assignments)
    descriptor_detail.to_csv(results_dir / "label_coverage_detail.csv", index=False)
    coverage_summary.to_csv(results_dir / "label_coverage_summary.csv", index=False)

    descriptor_summary, within_df, cross_df, semantic_summary = evaluate_semantic_consistency(assignments)
    descriptor_summary.to_csv(results_dir / "semantic_descriptor_assignments.csv", index=False)
    within_df.to_csv(results_dir / "semantic_consistency_within_primary.csv", index=False)
    cross_df.to_csv(results_dir / "semantic_consistency_cross_primary.csv", index=False)
    _write_json(results_dir / "semantic_consistency_summary.json", semantic_summary)
    _plot_semantic_matrix(within_df, cross_df, results_dir / "semantic_consistency_matrix.png")

    seeds = list(range(1001, 1001 + deletion_repetitions))
    deletion_summary, deletion_per_label = evaluate_label_deletion_stability(
        source_df,
        drop_fraction=0.2,
        seeds=seeds,
    )
    deletion_summary.to_csv(results_dir / "sea_label_deletion_stability_repetitions.csv", index=False)
    deletion_per_label.to_csv(results_dir / "sea_label_deletion_stability_per_label.csv", index=False)
    _write_json(
        results_dir / "sea_label_deletion_stability_summary.json",
        {
            "repetitions": deletion_repetitions,
            "drop_fraction": 0.2,
            "mean_label_cell_agreement": float(deletion_summary["label_cell_agreement"].mean()),
            "std_label_cell_agreement": float(deletion_summary["label_cell_agreement"].std(ddof=1)),
            "mean_molecule_exact_agreement": float(deletion_summary["molecule_exact_agreement"].mean()),
            "std_molecule_exact_agreement": float(deletion_summary["molecule_exact_agreement"].std(ddof=1)),
            "mean_positive_jaccard": float(deletion_summary["positive_jaccard"].mean()),
            "std_positive_jaccard": float(deletion_summary["positive_jaccard"].std(ddof=1)),
        },
    )


def _best_macro_f1_row(result: dict[str, object]) -> dict[str, object]:
    logs = pd.DataFrame(result["logs"])
    metric_column = (
        "val_optimized_macro_f1"
        if "val_optimized_macro_f1" in logs.columns
        else "val_macro_f1"
    )
    best_index = logs[metric_column].idxmax()
    row = logs.loc[best_index].to_dict()
    return {
        "metric": str(result["selection_metric"]),
        "best_epoch": int(row["epoch"]),
        "macro_f1": float(row[metric_column]),
    }


def _run_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig,
    local_molformer_path: Path | None,
) -> dict[str, object]:
    if model_name == "gnn":
        return train_gnn_baseline(train_df, test_df, config=config, labels=LABEL_COLUMNS)
    if model_name == "molformer":
        return train_molformer_baseline(
            train_df,
            test_df,
            config=config,
            labels=LABEL_COLUMNS,
            local_model_path=local_molformer_path,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def _dataset_summary(name: str, matrix: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    labels = matrix[LABEL_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        {
            "dataset": name,
            "rows": len(matrix),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "label": LABEL_COLUMNS,
            "positive_count": labels.sum().astype(int).reindex(LABEL_COLUMNS).to_numpy(),
            "missing_count": labels.isna().sum().astype(int).reindex(LABEL_COLUMNS).to_numpy(),
        }
    )


def _technical_validation_config(
    output_dir: Path,
    batch_size: int,
    epochs: int,
    training_seed: int,
) -> TrainingConfig:
    """Return the shared reviewer-validation configuration.

    The MolFormer settings intentionally follow codex_log/POM_baseline.ipynb.
    All runs use the same configuration so the reported Macro F1 values form
    a controlled technical validation, not a model-tuning exercise.
    """
    return TrainingConfig(
        nan_policy="drop",
        batch_size=batch_size,
        num_epochs=epochs,
        threshold=0.5,
        seed=training_seed,
        output_dir=str(output_dir),
        molformer_max_length=256,
        molformer_padding="max_length",
        molformer_attn_implementation="eager",
        l1_lambda=1e-8,
        mixed_precision=False,
        optimize_validation_thresholds=True,
        verbose=False,
    )


def run_controlled_comparison(
    results_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    training_seed: int,
) -> None:
    odornet_train = load_odornet("train", root=ROOT)
    odornet_test = load_odornet("test", root=ROOT)
    odornet_full = load_odornet("full", root=ROOT)
    gs_matrix, gs_source = prepare_gs_lf_matrix(ROOT / "data" / "raw" / "gs_lf.csv")
    gs_train, gs_test, gs_split_metadata = split_gs_lf_matrix(gs_matrix, seed=training_seed)

    pd.concat(
        [
            _dataset_summary("odornet", odornet_full, odornet_train, odornet_test),
            _dataset_summary("gs_lf", gs_matrix, gs_train, gs_test),
        ],
        ignore_index=True,
    ).to_csv(results_dir / "controlled_comparison_dataset_summary.csv", index=False)
    _write_json(
        results_dir / "gs_lf_preparation_summary.json",
        {
            "raw_rows": int(len(gs_source)),
            "raw_unique_smiles": int(gs_source["SMILES"].nunique()),
            "split_metadata": gs_split_metadata,
            "target_construction": "released SEA mapping followed by the Double-Drop rule",
        },
    )

    local_molformer_path = ROOT / "molformer_config"
    if not local_molformer_path.exists():
        local_molformer_path = None
    rows = []
    for dataset_name, train_df, test_df in (
        ("odornet", odornet_train, odornet_test),
        ("gs_lf", gs_train, gs_test),
    ):
        for model_name in ("gnn", "molformer"):
            config = _technical_validation_config(
                output_dir / "controlled_comparison" / dataset_name,
                batch_size=batch_size,
                epochs=epochs,
                training_seed=training_seed,
            )
            result = _run_model(
                model_name,
                train_df,
                test_df,
                config,
                local_molformer_path=local_molformer_path,
            )
            rows.append(
                {
                    "experiment": "controlled_comparison",
                    "dataset": dataset_name,
                    "model": model_name,
                    "training_seed": training_seed,
                    "nan_policy": config.nan_policy,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                    **_best_macro_f1_row(result),
                    "model_output_dir": result["output_dir"],
                    "model_source": result.get("model_source", ""),
                }
            )
    pd.DataFrame(rows).to_csv(results_dir / "controlled_comparison_metrics.csv", index=False)


def run_hierarchy_randomization(
    results_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    training_seed: int,
    repetitions: int,
) -> None:
    source_df = load_source_metadata(root=ROOT)
    base_mapping = load_main_label_mapping()
    base_matrix = build_double_drop_matrix(
        attach_main_labels(source_df, build_reverse_mapping(base_mapping))
    )
    released_train = load_odornet("train", root=ROOT)
    released_test = load_odornet("test", root=ROOT)
    local_molformer_path = ROOT / "molformer_config"
    if not local_molformer_path.exists():
        local_molformer_path = None

    metric_rows = []
    matrix_rows = []
    for repetition in range(1, repetitions + 1):
        mapping_seed = 2000 + repetition
        randomized_mapping = randomize_primary_secondary_mapping(base_mapping, seed=mapping_seed)
        randomized_matrix = build_double_drop_matrix(
            attach_main_labels(source_df, build_reverse_mapping(randomized_mapping))
        )
        merged = base_matrix[["SMILES", *LABEL_COLUMNS]].merge(
            randomized_matrix[["SMILES", *LABEL_COLUMNS]],
            on="SMILES",
            suffixes=("_base", "_randomized"),
            validate="one_to_one",
        )
        equal = np.ones(len(merged), dtype=bool)
        changed_cells = 0
        for label in LABEL_COLUMNS:
            baseline = pd.to_numeric(merged[f"{label}_base"], errors="coerce").to_numpy()
            randomized = pd.to_numeric(merged[f"{label}_randomized"], errors="coerce").to_numpy()
            current_equal = (baseline == randomized) | (np.isnan(baseline) & np.isnan(randomized))
            equal &= current_equal
            changed_cells += int((~current_equal).sum())
        matrix_rows.append(
            {
                "repetition": repetition,
                "mapping_seed": mapping_seed,
                "mapping_sha256": mapping_sha256(randomized_mapping),
                "changed_label_cells": changed_cells,
                "label_cell_change_rate": changed_cells / (len(merged) * len(LABEL_COLUMNS)),
                "molecule_exact_agreement": float(equal.mean()),
            }
        )

        train_df = released_train[["SMILES"]].merge(
            randomized_matrix,
            on="SMILES",
            how="left",
            validate="one_to_one",
        )
        test_df = released_test[["SMILES"]].merge(
            randomized_matrix,
            on="SMILES",
            how="left",
            validate="one_to_one",
        )
        for model_name in ("gnn", "molformer"):
            config = _technical_validation_config(
                output_dir / "hierarchy_randomization" / f"replicate_{repetition}",
                batch_size=batch_size,
                epochs=epochs,
                training_seed=training_seed,
            )
            result = _run_model(
                model_name,
                train_df,
                test_df,
                config,
                local_molformer_path=local_molformer_path,
            )
            metric_rows.append(
                {
                    "experiment": "hierarchy_randomization",
                    "repetition": repetition,
                    "mapping_seed": mapping_seed,
                    "mapping_sha256": mapping_sha256(randomized_mapping),
                    "model": model_name,
                    "training_seed": training_seed,
                    "nan_policy": config.nan_policy,
                    "batch_size": batch_size,
                    "num_epochs": epochs,
                    **_best_macro_f1_row(result),
                    "model_output_dir": result["output_dir"],
                    "model_source": result.get("model_source", ""),
                }
            )
    pd.DataFrame(matrix_rows).to_csv(results_dir / "hierarchy_randomization_matrix_disruption.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(results_dir / "hierarchy_randomization_metrics.csv", index=False)
    (
        metrics.groupby("model", as_index=False)
        .agg(
            repetitions=("repetition", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .to_csv(results_dir / "hierarchy_randomization_summary.csv", index=False)
    )


def run_stereo_safe_split_training(
    results_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    training_seed: int,
) -> None:
    full_df = load_odornet("full", root=ROOT)
    train_df, validation_df, search_df, split_metadata = search_stereo_safe_split(
        full_df,
        test_size=0.2,
        seed_candidates=range(1000),
        target_labels=LABEL_COLUMNS,
    )

    search_df.to_csv(results_dir / "stereo_safe_split_seed_search.csv", index=False)
    split_rows = pd.concat(
        [
            train_df.assign(split="train"),
            validation_df.assign(split="validation"),
        ],
        ignore_index=True,
    )
    split_rows.to_csv(results_dir / "stereo_safe_split_assignments.csv", index=False)
    _dataset_summary("stereo_safe_split", full_df, train_df, validation_df).to_csv(
        results_dir / "stereo_safe_split_dataset_summary.csv",
        index=False,
    )
    _write_json(
        results_dir / "stereo_safe_split_summary.json",
        {
            **split_metadata,
            "seed_search_table": "stereo_safe_split_seed_search.csv",
            "assignment_table": "stereo_safe_split_assignments.csv",
            "dataset_summary_table": "stereo_safe_split_dataset_summary.csv",
        },
    )

    local_molformer_path = ROOT / "molformer_config"
    if not local_molformer_path.exists():
        local_molformer_path = None

    rows = []
    for model_name in ("gnn", "molformer"):
        config = _technical_validation_config(
            output_dir / "stereo_safe_split",
            batch_size=batch_size,
            epochs=epochs,
            training_seed=training_seed,
        )
        result = _run_model(
            model_name,
            train_df,
            validation_df,
            config,
            local_molformer_path=local_molformer_path,
        )
        rows.append(
            {
                "experiment": "stereo_safe_split",
                "dataset": "odornet",
                "model": model_name,
                "training_seed": training_seed,
                "nan_policy": config.nan_policy,
                "batch_size": batch_size,
                "num_epochs": epochs,
                **_best_macro_f1_row(result),
                "model_output_dir": result["output_dir"],
                "model_source": result.get("model_source", ""),
            }
        )
    pd.DataFrame(rows).to_csv(results_dir / "stereo_safe_split_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--training-seed", type=int, default=959)
    parser.add_argument("--deletion-repetitions", type=int, default=10)
    parser.add_argument("--hierarchy-repetitions", type=int, default=3)
    parser.add_argument(
        "--stage",
        choices=("all", "taxonomy", "comparison", "hierarchy", "stereo_safe"),
        default="all",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.results_dir / "reviewer_evaluation_config.json",
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "training_seed": args.training_seed,
            "deletion_repetitions": args.deletion_repetitions,
            "hierarchy_repetitions": args.hierarchy_repetitions,
            "stage_requested": args.stage,
            "stage_scope_note": (
                "Use --stage all for a single invocation that regenerates every "
                "taxonomy, comparison, and hierarchy output. This file records "
                "the requested stage for the latest command-line invocation."
            ),
            "technical_validation_metric": (
                "validation_optimized_macro_f1_with_per_label_thresholds"
            ),
            "molformer_protocol": {
                "reference_notebook": "codex_log/POM_baseline.ipynb",
                "max_length": 256,
                "padding": "max_length",
                "attention_implementation": "eager",
                "mixed_precision": False,
                "l1_lambda": 1e-8,
            },
        },
    )
    if args.stage in {"all", "taxonomy"}:
        run_taxonomy_audits(args.results_dir, args.deletion_repetitions)
    if args.stage in {"all", "comparison"}:
        run_controlled_comparison(
            args.results_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            training_seed=args.training_seed,
        )
    if args.stage in {"all", "hierarchy"}:
        run_hierarchy_randomization(
            args.results_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            training_seed=args.training_seed,
            repetitions=args.hierarchy_repetitions,
        )
    if args.stage in {"all", "stereo_safe"}:
        run_stereo_safe_split_training(
            args.results_dir,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            training_seed=args.training_seed,
        )


if __name__ == "__main__":
    main()
