"""Baseline training utilities following the original OdorNet notebook.

The original reference notebook defined two baseline families:
- MolFormer fine-tuning for SMILES multi-label classification.
- A simple two-layer GCN baseline using RDKit molecular graphs.

This module keeps those modeling choices while simplifying metric logging and
using repository-relative data inputs.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .datasets import LABEL_COLUMNS


DEFAULT_MOLFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"


@dataclass
class TrainingConfig:
    nan_policy: str = "drop"
    batch_size: int = 48
    num_epochs: int = 30
    threshold: float = 0.5
    seed: int = 959
    output_dir: str = "outputs/baseline"


def set_seed(seed: int = 959) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def target_columns(df: pd.DataFrame) -> list[str]:
    return [label for label in LABEL_COLUMNS if label in df.columns]


def calculate_pos_weights(df: pd.DataFrame, labels: list[str]) -> torch.Tensor:
    weights = []
    for label in labels:
        valid = pd.to_numeric(df[label], errors="coerce").dropna()
        n_pos = (valid == 1.0).sum()
        n_neg = (valid == 0.0).sum()
        weights.append(float(n_neg / n_pos) if n_pos > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedMaskedBCELoss(nn.Module):
    def __init__(self, nan_policy: str, pos_weight: torch.Tensor | None = None):
        super().__init__()
        if nan_policy not in {"drop", "union", "intersection"}:
            raise ValueError("nan_policy must be one of: drop, union, intersection")
        self.nan_policy = nan_policy
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1 and logits.dim() == 2:
            targets = targets.unsqueeze(1)

        if self.nan_policy == "drop":
            targets_filled = torch.nan_to_num(targets, nan=0.0)
            mask = ~torch.isnan(targets)
        elif self.nan_policy == "union":
            targets_filled = torch.nan_to_num(targets, nan=1.0)
            mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            targets_filled = torch.nan_to_num(targets, nan=0.0)
            mask = torch.ones_like(targets, dtype=torch.bool)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets_filled,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        valid_count = mask.sum()
        if valid_count == 0:
            return logits.sum() * 0.0
        return (loss * mask.float()).sum() / valid_count


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    nan_policy: str,
    threshold: float = 0.5,
    labels: list[str] | None = None,
) -> dict[str, object]:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_prob = y_prob.reshape(-1, 1)

    per_label = []
    f1_values = []
    auroc_values = []
    labels = labels or [str(i) for i in range(y_true.shape[1])]

    for idx, label in enumerate(labels):
        truth = y_true[:, idx]
        prob = y_prob[:, idx]
        mask = ~np.isnan(truth)
        if nan_policy == "union":
            truth = np.nan_to_num(truth, nan=1.0)
            mask = np.ones_like(truth, dtype=bool)
        elif nan_policy == "intersection":
            truth = np.nan_to_num(truth, nan=0.0)
            mask = np.ones_like(truth, dtype=bool)

        if mask.sum() == 0:
            f1 = 0.0
            auroc = 0.5
            support = 0
            positives = 0
        else:
            truth_v = truth[mask].astype(int)
            prob_v = prob[mask]
            pred_v = (prob_v >= threshold).astype(int)
            f1 = float(f1_score(truth_v, pred_v, zero_division=0))
            auroc = float(roc_auc_score(truth_v, prob_v)) if len(np.unique(truth_v)) > 1 else 0.5
            support = int(mask.sum())
            positives = int(truth_v.sum())
        f1_values.append(f1)
        auroc_values.append(auroc)
        per_label.append(
            {
                "label": label,
                "support": support,
                "positives": positives,
                "f1": f1,
                "auroc": auroc,
            }
        )

    if nan_policy == "drop":
        global_mask = ~np.isnan(y_true)
        global_truth = y_true[global_mask].astype(int)
        global_prob = y_prob[global_mask]
    elif nan_policy == "union":
        global_truth = np.nan_to_num(y_true, nan=1.0).astype(int).ravel()
        global_prob = y_prob.ravel()
    else:
        global_truth = np.nan_to_num(y_true, nan=0.0).astype(int).ravel()
        global_prob = y_prob.ravel()

    global_pred = (global_prob >= threshold).astype(int)
    return {
        "macro_f1": float(np.mean(f1_values)),
        "micro_f1": float(f1_score(global_truth, global_pred, zero_division=0)),
        "macro_auroc": float(np.mean(auroc_values)),
        "micro_auroc": float(roc_auc_score(global_truth, global_prob))
        if len(np.unique(global_truth)) > 1
        else 0.5,
        "per_label": per_label,
    }


def _epoch_summary(prefix: str, loss: float, metrics: dict[str, object]) -> dict[str, float]:
    return {
        f"{prefix}_loss": float(loss),
        f"{prefix}_macro_f1": float(metrics["macro_f1"]),
        f"{prefix}_micro_f1": float(metrics["micro_f1"]),
        f"{prefix}_macro_auroc": float(metrics["macro_auroc"]),
        f"{prefix}_micro_auroc": float(metrics["micro_auroc"]),
    }


class SmilesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        labels: list[str],
        smiles_col: str = "SMILES",
        max_length: int = 256,
    ):
        self.smiles = df[smiles_col].tolist()
        self.targets = torch.tensor(
            df[labels].apply(pd.to_numeric, errors="coerce").values,
            dtype=torch.float32,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = self.targets[idx]
        return item


class MolFormerMLP(nn.Module):
    def __init__(
        self,
        molformer_base: nn.Module,
        num_labels: int,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.molformer_base = molformer_base
        hidden_size = molformer_base.config.hidden_size
        layer_sizes = [hidden_size, *(hidden_layers or [512, 384, 256])]

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.final_layer = nn.Linear(layer_sizes[-1], num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.molformer_base(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if attention_mask is not None:
            denom = attention_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
            pooled = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        else:
            pooled = hidden_states.mean(dim=1)
        embedding = self.mlp(pooled)
        logits = self.final_layer(embedding)
        return embedding, logits


def resolve_molformer_source(
    model_name: str = DEFAULT_MOLFORMER_MODEL,
    local_model_path: str | Path | None = None,
) -> tuple[str, bool]:
    """Resolve MolFormer source and whether transformers should stay offline."""
    if local_model_path is None:
        return model_name, False

    local_path = Path(local_model_path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(f"Local MolFormer path does not exist: {local_path}")
    return str(local_path), True


def load_molformer(
    model_name: str = DEFAULT_MOLFORMER_MODEL,
    local_model_path: str | Path | None = None,
):
    """Load MolFormer from a local folder when provided, otherwise from Hugging Face."""
    from transformers import AutoModel, AutoTokenizer

    source, local_files_only = resolve_molformer_source(
        model_name=model_name,
        local_model_path=local_model_path,
    )
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    try:
        base = AutoModel.from_pretrained(
            source,
            deterministic_eval=False,
            **load_kwargs,
        )
    except TypeError:
        base = AutoModel.from_pretrained(source, **load_kwargs)
    return tokenizer, base


def train_molformer_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig | None = None,
    model_name: str = DEFAULT_MOLFORMER_MODEL,
    local_model_path: str | Path | None = None,
    labels: list[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    config = config or TrainingConfig()
    labels = labels or target_columns(train_df)
    set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_source, _ = resolve_molformer_source(
        model_name=model_name,
        local_model_path=local_model_path,
    )
    tokenizer, base = load_molformer(
        model_name=model_name,
        local_model_path=local_model_path,
    )
    model = MolFormerMLP(base, num_labels=len(labels)).to(device)
    train_loader = DataLoader(
        SmilesDataset(train_df, tokenizer, labels),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SmilesDataset(test_df, tokenizer, labels),
        batch_size=config.batch_size,
        shuffle=False,
    )

    pos_weight = calculate_pos_weights(train_df, labels).to(device)
    criterion = WeightedMaskedBCELoss(config.nan_policy, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    result = _train_torch_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        labels,
        config,
        device,
        model_name="molformer",
    )
    result["model_source"] = model_source
    return result


def _collect_probs_and_loss(model, loader, criterion, device, is_graph: bool = False):
    total_loss = 0.0
    probs = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if is_graph:
                batch = batch.to(device)
                logits = model(batch)
                y = batch.y
            else:
                y = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                _, logits = model(**batch)
            loss = criterion(logits, y)
            total_loss += loss.item()
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    return total_loss / max(len(loader), 1), np.vstack(targets), np.vstack(probs)


def _train_torch_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    labels,
    config,
    device,
    model_name: str,
    is_graph: bool = False,
) -> dict[str, object]:
    output_dir = Path(config.output_dir) / model_name / config.nan_policy
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    best_macro_f1 = -1.0
    best_per_label = None

    for epoch in tqdm(range(config.num_epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0.0
        train_probs = []
        train_targets = []
        for batch in train_loader:
            optimizer.zero_grad()
            if is_graph:
                batch = batch.to(device)
                logits = model(batch)
                y = batch.y
            else:
                y = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                _, logits = model(**batch)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())

        train_loss = train_loss / max(len(train_loader), 1)
        train_metrics = compute_metrics(
            np.vstack(train_targets),
            np.vstack(train_probs),
            config.nan_policy,
            config.threshold,
            labels,
        )
        val_loss, val_targets, val_probs = _collect_probs_and_loss(
            model, val_loader, criterion, device, is_graph=is_graph
        )
        val_metrics = compute_metrics(
            val_targets,
            val_probs,
            config.nan_policy,
            config.threshold,
            labels,
        )

        row = {
            "epoch": epoch + 1,
            **_epoch_summary("train", train_loss, train_metrics),
            **_epoch_summary("val", val_loss, val_metrics),
        }
        logs.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_per_label = val_metrics["per_label"]
            torch.save(model.state_dict(), output_dir / "best_val_macrof1.pt")

    with (output_dir / "training_logs.json").open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), "logs": logs, "best_per_label": best_per_label}, handle, indent=2)

    if best_per_label is not None:
        pd.DataFrame(best_per_label).to_csv(output_dir / "best_per_label_metrics.csv", index=False)

    return {"logs": logs, "best_macro_f1": best_macro_f1, "best_per_label": best_per_label, "output_dir": str(output_dir)}


def train_gnn_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig | None = None,
    labels: list[str] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    from rdkit import Chem, RDLogger
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool

    RDLogger.DisableLog("rdApp.warning")
    config = config or TrainingConfig()
    labels = labels or target_columns(train_df)
    set_seed(config.seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    allowed_atoms = [
        "C",
        "N",
        "O",
        "S",
        "F",
        "Cl",
        "Br",
        "I",
        "P",
        "H",
        "B",
        "Si",
        "Se",
        "Te",
        "As",
        "Unknown",
    ]

    def atom_features(atom):
        symbol = atom.GetSymbol() if atom.GetSymbol() in allowed_atoms else "Unknown"
        return [float(symbol == item) for item in allowed_atoms]

    def smiles_to_graph(smiles: str) -> Data:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return Data(
                x=torch.zeros((1, len(allowed_atoms))),
                edge_index=torch.empty((2, 0), dtype=torch.long),
            )
        x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
        edges = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.extend([[i, j], [j, i]])
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        return Data(x=x, edge_index=edge_index)

    class MoleculeGraphDataset(Dataset):
        def __init__(self, df):
            self.labels = torch.tensor(
                df[labels].apply(pd.to_numeric, errors="coerce").values,
                dtype=torch.float32,
            )
            self.graphs = []
            for idx, smiles in enumerate(tqdm(df["SMILES"], desc="Building molecular graphs")):
                graph = smiles_to_graph(smiles)
                graph.y = self.labels[idx].unsqueeze(0)
                self.graphs.append(graph)

        def __len__(self):
            return len(self.graphs)

        def __getitem__(self, idx):
            return self.graphs[idx]

    class SimpleGNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_labels, dropout=0.2):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_labels),
            )

        def forward(self, data):
            x = F.relu(self.conv1(data.x, data.edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, data.edge_index))
            x = global_mean_pool(x, data.batch)
            return self.classifier(x)

    train_loader = PyGDataLoader(
        MoleculeGraphDataset(train_df), batch_size=config.batch_size, shuffle=True
    )
    val_loader = PyGDataLoader(
        MoleculeGraphDataset(test_df), batch_size=config.batch_size, shuffle=False
    )
    model = SimpleGNN(
        input_dim=len(allowed_atoms),
        hidden_dim=256,
        num_labels=len(labels),
        dropout=0.2,
    ).to(device)
    pos_weight = calculate_pos_weights(train_df, labels).to(device)
    criterion = WeightedMaskedBCELoss(config.nan_policy, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    return _train_torch_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        labels,
        config,
        device,
        model_name="gnn",
        is_graph=True,
    )
