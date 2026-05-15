from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src import config
from src.core.classification.contradiction import get_nli_pipeline
from src.ui.services import get_nli_batch_size


DATA_PATH = Path("data/golden_nli_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/nli_eval/golden_nli")
LABELS = ["entailment", "neutral", "contradiction"]
PLOT_COLORS = {
    "entailment": "#2f9e44",
    "neutral": "#495057",
    "contradiction": "#d94841",
}


def normalize_label(label: object) -> str:
    value = str(label).strip().lower()
    aliases = {
        "label_0": "entailment",
        "label_1": "contradiction",
        "label_2": "neutral",
    }
    return aliases.get(value, value)


def scores_to_dict(scores: list[dict]) -> dict[str, float]:
    return {
        normalize_label(item["label"]): float(item["score"])
        for item in scores
    }


def batch_outputs(model, pairs: list[tuple[str, str]], batch_size: int) -> list[list[dict]]:
    inputs = [
        {"text": text, "text_pair": text_pair}
        for text, text_pair in pairs
    ]
    return model(
        inputs,
        batch_size=batch_size,
        truncation=True,
        max_length=512,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the current NLI model on golden_nli_dataset.csv.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to the golden NLI CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, predictions, errors, and plots.",
    )
    return parser.parse_args()


def build_probabilities(
    forward: list[dict[str, float]],
    backward: list[dict[str, float]],
) -> np.ndarray:
    return np.array(
        [
            [
                (forward_scores.get(label, 0.0) + backward_scores.get(label, 0.0)) / 2.0
                for label in LABELS
            ]
            for forward_scores, backward_scores in zip(forward, backward, strict=True)
        ],
        dtype=float,
    )


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
) -> dict:
    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    y_true_binary = np.array(
        [
            [1 if true_label == label else 0 for label in LABELS]
            for true_label in y_true
        ],
        dtype=int,
    )

    per_class = {}
    per_class_auc = []
    support = Counter(y_true)
    prediction_counts = Counter(y_pred)
    for index, label in enumerate(LABELS):
        try:
            auc_value = float(roc_auc_score(y_true_binary[:, index], probabilities[:, index]))
            per_class_auc.append(auc_value)
        except ValueError:
            auc_value = None

        per_class[label] = {
            "f1": float(class_f1[index]),
            "roc_auc_ovr": auc_value,
            "support": int(support[label]),
            "predicted": int(prediction_counts[label]),
        }

    return {
        "dataset": str(DATA_PATH),
        "model": str(getattr(config, "nli_model", "")),
        "mode": "bidirectional_average_argmax",
        "labels": LABELS,
        "row_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "roc_auc_macro_ovr": (
            float(np.mean(per_class_auc))
            if per_class_auc
            else None
        ),
        "per_class": per_class,
        "support": {label: int(support[label]) for label in LABELS},
        "prediction_counts": {
            label: int(prediction_counts[label])
            for label in LABELS
        },
        "confusion_matrix": {
            "rows_true_cols_pred": matrix.astype(int).tolist(),
            "labels": LABELS,
        },
        "total_errors": int(sum(true != pred for true, pred in zip(y_true, y_pred, strict=True))),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_predictions_csv(
    path: Path,
    rows: list[dict[str, str]],
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    *,
    only_errors: bool = False,
) -> None:
    fields = [
        "id",
        "tk_chunk_id",
        "true_label",
        "pred_label",
        "correct",
        "prob_entailment",
        "prob_neutral",
        "prob_contradiction",
        "contract_clause",
        "tk_evidence",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, true_label, pred_label, scores in zip(
            rows,
            y_true,
            y_pred,
            probabilities,
            strict=True,
        ):
            correct = true_label == pred_label
            if only_errors and correct:
                continue
            writer.writerow(
                {
                    "id": row["id"],
                    "tk_chunk_id": row["tk_chunk_id"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "correct": int(correct),
                    "prob_entailment": f"{scores[0]:.8f}",
                    "prob_neutral": f"{scores[1]:.8f}",
                    "prob_contradiction": f"{scores[2]:.8f}",
                    "contract_clause": row["contract_clause"],
                    "tk_evidence": row["tk_evidence"],
                }
            )


def save_reports(
    output_dir: Path,
    rows: list[dict[str, str]],
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    metrics: dict,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_json": output_dir / "metrics.json",
        "predictions_csv": output_dir / "predictions.csv",
        "errors_csv": output_dir / "errors.csv",
    }
    write_json(paths["metrics_json"], metrics)
    write_predictions_csv(paths["predictions_csv"], rows, y_true, y_pred, probabilities)
    write_predictions_csv(
        paths["errors_csv"],
        rows,
        y_true,
        y_pred,
        probabilities,
        only_errors=True,
    )
    return {name: str(path) for name, path in paths.items()}


def save_plots(
    output_dir: Path,
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    metrics: dict,
) -> dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib_unavailable_using_pillow=({exc})")
        return save_plots_with_pillow(output_dir, y_true, y_pred, probabilities, metrics)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    colors = [PLOT_COLORS[label] for label in LABELS]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#343a40",
            "axes.labelcolor": "#212529",
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.color": "#343a40",
            "ytick.color": "#343a40",
            "grid.color": "#dee2e6",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )

    def save_figure(fig, name: str) -> str:
        path = plot_dir / f"{name}.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return str(path)

    matrix = np.array(metrics["confusion_matrix"]["rows_true_cols_pred"], dtype=int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(matrix_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.tick_params(length=0)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            pct = matrix_pct[row_index, col_index] * 100
            text_color = "white" if matrix_pct[row_index, col_index] >= 0.45 else "#212529"
            ax.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]}\n{pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
                fontweight="bold" if row_index == col_index else "normal",
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share within true class")
    fig.tight_layout()
    saved["confusion_matrix"] = save_figure(fig, "confusion_matrix")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    f1_values = [metrics["per_class"][label]["f1"] for label in LABELS]
    bars = ax.bar(LABELS, f1_values, color=colors, width=0.58)
    ax.axhline(
        metrics["macro_f1"],
        color="#1864ab",
        linestyle="--",
        linewidth=1.6,
        label=f"macro F1 = {metrics['macro_f1']:.3f}",
    )
    ax.set_ylim(0, 1)
    ax.set_title("F1 Score by Class")
    ax.set_ylabel("F1")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in f1_values], padding=4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    saved["f1_by_class"] = save_figure(fig, "f1_by_class")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    auc_values = [
        metrics["per_class"][label]["roc_auc_ovr"] or 0.0
        for label in LABELS
    ]
    bars = ax.bar(LABELS, auc_values, color=colors, width=0.58)
    macro_auc = metrics.get("roc_auc_macro_ovr")
    if macro_auc is not None:
        ax.axhline(
            macro_auc,
            color="#1864ab",
            linestyle="--",
            linewidth=1.6,
            label=f"macro ROC-AUC = {macro_auc:.3f}",
        )
    ax.set_ylim(0, 1)
    ax.set_title("ROC-AUC One-vs-Rest by Class")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in auc_values], padding=4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    saved["roc_auc_by_class"] = save_figure(fig, "roc_auc_by_class")

    y_true_binary = np.array(
        [
            [1 if true_label == label else 0 for label in LABELS]
            for true_label in y_true
        ],
        dtype=int,
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot([0, 1], [0, 1], color="#adb5bd", linestyle="--", linewidth=1.2)
    for index, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true_binary[:, index], probabilities[:, index])
        auc_value = metrics["per_class"][label]["roc_auc_ovr"]
        auc_text = "nan" if auc_value is None else f"{auc_value:.3f}"
        ax.plot(
            fpr,
            tpr,
            label=f"{label} AUC={auc_text}",
            color=PLOT_COLORS[label],
            linewidth=2,
        )
    ax.set_title("ROC Curves One-vs-Rest")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    saved["roc_curves"] = save_figure(fig, "roc_curves")

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    x = np.arange(len(LABELS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    true_bars = ax.bar(
        x - width / 2,
        [true_counts[label] for label in LABELS],
        width,
        label="true",
        color="#74c0fc",
    )
    pred_bars = ax.bar(
        x + width / 2,
        [pred_counts[label] for label in LABELS],
        width,
        label="predicted",
        color="#ffa94d",
    )
    ax.set_title("True vs Predicted Label Counts")
    ax.set_ylabel("Count")
    ax.set_xticks(x, LABELS)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(true_bars, padding=3)
    ax.bar_label(pred_bars, padding=3)
    ax.legend()
    fig.tight_layout()
    saved["label_counts"] = save_figure(fig, "label_counts")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    metric_names = ["F1", "ROC-AUC"]
    metric_x = np.arange(len(metric_names))
    grouped_width = 0.22
    for index, label in enumerate(LABELS):
        values = [
            metrics["per_class"][label]["f1"],
            metrics["per_class"][label]["roc_auc_ovr"] or 0.0,
        ]
        offset = (index - 1) * grouped_width
        bars = ax.bar(
            metric_x + offset,
            values,
            grouped_width,
            label=label,
            color=PLOT_COLORS[label],
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    ax.set_title("Metric Comparison by Class")
    ax.set_ylabel("Score")
    ax.set_xticks(metric_x, metric_names)
    ax.set_ylim(0, 1)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    saved["metric_comparison"] = save_figure(fig, "metric_comparison")

    max_probabilities = probabilities.max(axis=1)
    is_correct = np.array(
        [true == pred for true, pred in zip(y_true, y_pred, strict=True)],
        dtype=bool,
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    bins = np.linspace(0, 1, 12)
    ax.hist(
        max_probabilities[is_correct],
        bins=bins,
        alpha=0.72,
        label=f"correct ({int(is_correct.sum())})",
        color="#2f9e44",
        edgecolor="white",
    )
    ax.hist(
        max_probabilities[~is_correct],
        bins=bins,
        alpha=0.72,
        label=f"errors ({int((~is_correct).sum())})",
        color="#d94841",
        edgecolor="white",
    )
    ax.set_title("Prediction Confidence by Outcome")
    ax.set_xlabel("Max predicted probability")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.tight_layout()
    saved["confidence_by_outcome"] = save_figure(fig, "confidence_by_outcome")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for index, (ax, label) in enumerate(zip(axes, LABELS, strict=True)):
        true_mask = np.array([true == label for true in y_true], dtype=bool)
        ax.hist(
            probabilities[true_mask, index],
            bins=np.linspace(0, 1, 11),
            alpha=0.78,
            color=PLOT_COLORS[label],
            edgecolor="white",
        )
        ax.set_title(label)
        ax.set_xlabel(f"P({label})")
        ax.set_xlim(0, 1)
        ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Count for true class")
    fig.suptitle("True-Class Probability Distributions", fontweight="bold", y=1.03)
    fig.tight_layout()
    saved["true_class_probability_distributions"] = save_figure(
        fig,
        "true_class_probability_distributions",
    )

    return saved


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) for index in (0, 2, 4))


def text_size(draw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(draw, xy: tuple[float, float], text: str, font, fill) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def save_pillow_image(image, path: Path) -> str:
    image.save(path)
    return str(path)


def save_plots_with_pillow(
    output_dir: Path,
    y_true: list[str],
    y_pred: list[str],
    probabilities: np.ndarray,
    metrics: dict,
) -> dict[str, str]:
    from PIL import Image, ImageDraw, ImageFont

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    font = ImageFont.load_default()
    background = (255, 255, 255)
    axis_color = (52, 58, 64)
    grid_color = (222, 226, 230)

    matrix = np.array(metrics["confusion_matrix"]["rows_true_cols_pred"], dtype=int)
    cell = 112
    left = 180
    top = 90
    width = left + cell * len(LABELS) + 40
    height = top + cell * len(LABELS) + 80
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.text((24, 24), "Confusion matrix", font=font, fill=axis_color)
    draw.text((left + 80, 58), "Predicted label", font=font, fill=axis_color)
    draw.text((24, top + 115), "True label", font=font, fill=axis_color)
    max_value = max(1, int(matrix.max()))
    for col_index, label in enumerate(LABELS):
        draw_centered_text(
            draw,
            (left + col_index * cell + cell / 2, top - 22),
            label,
            font,
            axis_color,
        )
    for row_index, label in enumerate(LABELS):
        draw.text((30, top + row_index * cell + cell / 2 - 6), label, font=font, fill=axis_color)
        for col_index in range(len(LABELS)):
            value = int(matrix[row_index, col_index])
            intensity = int(255 - 155 * value / max_value)
            fill = (intensity, intensity + 12, 255)
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline=background)
            draw_centered_text(
                draw,
                (x0 + cell / 2, y0 + cell / 2),
                str(value),
                font,
                (0, 0, 0),
            )
    saved["confusion_matrix"] = save_pillow_image(
        image,
        plot_dir / "confusion_matrix.png",
    )

    def draw_bar_chart(
        title: str,
        values: list[float],
        ylabel: str,
        path: Path,
        *,
        max_y: float = 1.0,
        reference: float | None = None,
        reference_label: str = "",
    ) -> str:
        image = Image.new("RGB", (820, 520), background)
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = 90, 70, 780, 430
        draw.text((24, 24), title, font=font, fill=axis_color)
        draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
        draw.line((left, top, left, bottom), fill=axis_color, width=2)
        draw.text((24, 230), ylabel, font=font, fill=axis_color)
        for tick in range(6):
            y_value = max_y * tick / 5
            y = bottom - (bottom - top) * y_value / max_y
            draw.line((left - 4, y, right, y), fill=grid_color, width=1)
            draw.text((42, y - 6), f"{y_value:.1f}", font=font, fill=axis_color)
        if reference is not None:
            ref_y = bottom - (bottom - top) * reference / max_y
            draw.line((left, ref_y, right, ref_y), fill=(24, 100, 171), width=2)
            draw.text((right - 120, ref_y - 18), reference_label, font=font, fill=(24, 100, 171))
        slot = (right - left) / len(LABELS)
        bar_width = slot * 0.55
        for index, (label, value) in enumerate(zip(LABELS, values, strict=True)):
            x_center = left + slot * index + slot / 2
            bar_top = bottom - (bottom - top) * value / max_y
            color = hex_to_rgb(PLOT_COLORS[label])
            draw.rectangle(
                (x_center - bar_width / 2, bar_top, x_center + bar_width / 2, bottom),
                fill=color,
            )
            draw_centered_text(draw, (x_center, bar_top - 14), f"{value:.3f}", font, axis_color)
            draw_centered_text(draw, (x_center, bottom + 22), label, font, axis_color)
        return save_pillow_image(image, path)

    f1_values = [metrics["per_class"][label]["f1"] for label in LABELS]
    saved["f1_by_class"] = draw_bar_chart(
        "F1 by class",
        f1_values,
        "F1",
        plot_dir / "f1_by_class.png",
        reference=metrics["macro_f1"],
        reference_label=f"macro {metrics['macro_f1']:.3f}",
    )

    auc_values = [
        metrics["per_class"][label]["roc_auc_ovr"] or 0.0
        for label in LABELS
    ]
    macro_auc = metrics.get("roc_auc_macro_ovr") or 0.0
    saved["roc_auc_by_class"] = draw_bar_chart(
        "ROC-AUC one-vs-rest by class",
        auc_values,
        "ROC-AUC",
        plot_dir / "roc_auc_by_class.png",
        reference=macro_auc,
        reference_label=f"macro {macro_auc:.3f}",
    )

    y_true_binary = np.array(
        [
            [1 if true_label == label else 0 for label in LABELS]
            for true_label in y_true
        ],
        dtype=int,
    )
    image = Image.new("RGB", (720, 620), background)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 80, 70, 650, 530
    draw.text((24, 24), "ROC curves one-vs-rest", font=font, fill=axis_color)
    draw.rectangle((left, top, right, bottom), outline=axis_color, width=2)
    for tick in range(6):
        x = left + (right - left) * tick / 5
        y = bottom - (bottom - top) * tick / 5
        draw.line((x, top, x, bottom), fill=grid_color, width=1)
        draw.line((left, y, right, y), fill=grid_color, width=1)
        draw.text((x - 8, bottom + 10), f"{tick / 5:.1f}", font=font, fill=axis_color)
        draw.text((38, y - 6), f"{tick / 5:.1f}", font=font, fill=axis_color)
    draw.line((left, bottom, right, top), fill=(173, 181, 189), width=1)
    draw.text((300, 575), "False positive rate", font=font, fill=axis_color)
    draw.text((12, 292), "TPR", font=font, fill=axis_color)
    legend_y = 92
    for index, label in enumerate(LABELS):
        fpr, tpr, _ = roc_curve(y_true_binary[:, index], probabilities[:, index])
        points = [
            (
                left + (right - left) * float(x_value),
                bottom - (bottom - top) * float(y_value),
            )
            for x_value, y_value in zip(fpr, tpr, strict=True)
        ]
        color = hex_to_rgb(PLOT_COLORS[label])
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        auc_value = metrics["per_class"][label]["roc_auc_ovr"]
        auc_text = "nan" if auc_value is None else f"{auc_value:.3f}"
        draw.line((470, legend_y, 510, legend_y), fill=color, width=3)
        draw.text((520, legend_y - 6), f"{label} AUC={auc_text}", font=font, fill=axis_color)
        legend_y += 22
    saved["roc_curves"] = save_pillow_image(image, plot_dir / "roc_curves.png")

    true_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    image = Image.new("RGB", (820, 520), background)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 780, 430
    draw.text((24, 24), "True vs predicted label counts", font=font, fill=axis_color)
    draw.line((left, bottom, right, bottom), fill=axis_color, width=2)
    draw.line((left, top, left, bottom), fill=axis_color, width=2)
    max_count = max(1, max(true_counts.values()), max(pred_counts.values()))
    for tick in range(6):
        y_value = max_count * tick / 5
        y = bottom - (bottom - top) * y_value / max_count
        draw.line((left - 4, y, right, y), fill=grid_color, width=1)
        draw.text((42, y - 6), f"{y_value:.0f}", font=font, fill=axis_color)
    slot = (right - left) / len(LABELS)
    bar_width = slot * 0.28
    for index, label in enumerate(LABELS):
        x_center = left + slot * index + slot / 2
        true_height = (bottom - top) * true_counts[label] / max_count
        pred_height = (bottom - top) * pred_counts[label] / max_count
        draw.rectangle(
            (x_center - bar_width - 4, bottom - true_height, x_center - 4, bottom),
            fill=(116, 192, 252),
        )
        draw.rectangle(
            (x_center + 4, bottom - pred_height, x_center + bar_width + 4, bottom),
            fill=(255, 169, 77),
        )
        draw_centered_text(draw, (x_center, bottom + 22), label, font, axis_color)
    draw.rectangle((610, 88, 628, 106), fill=(116, 192, 252))
    draw.text((636, 90), "true", font=font, fill=axis_color)
    draw.rectangle((610, 116, 628, 134), fill=(255, 169, 77))
    draw.text((636, 118), "predicted", font=font, fill=axis_color)
    saved["label_counts"] = save_pillow_image(image, plot_dir / "label_counts.png")

    return saved


def main() -> None:
    args = parse_args()
    data_path = args.data
    output_dir = args.output_dir
    rows = list(csv.DictReader(data_path.open(encoding="utf-8-sig")))
    y_true = [normalize_label(row["label"]) for row in rows]
    forward_pairs = [
        (row["contract_clause"].strip(), row["tk_evidence"].strip())
        for row in rows
    ]
    backward_pairs = [
        (text_pair, text)
        for text, text_pair in forward_pairs
    ]

    print(f"rows={len(rows)} labels={dict(Counter(y_true))}", flush=True)
    print("loading_model...", flush=True)
    model = get_nli_pipeline()
    batch_size = get_nli_batch_size()
    print(f"batch_size={batch_size}", flush=True)

    print("scoring_forward...", flush=True)
    forward = [
        scores_to_dict(scores)
        for scores in batch_outputs(model, forward_pairs, batch_size)
    ]
    print("scoring_backward...", flush=True)
    backward = [
        scores_to_dict(scores)
        for scores in batch_outputs(model, backward_pairs, batch_size)
    ]

    probabilities = build_probabilities(forward, backward)
    y_pred = [
        LABELS[int(index)]
        for index in np.argmax(probabilities, axis=1)
    ]
    metrics = compute_metrics(y_true, y_pred, probabilities)
    metrics["dataset"] = str(data_path)
    report_paths = save_reports(
        output_dir,
        rows,
        y_true,
        y_pred,
        probabilities,
        metrics,
    )
    plot_paths = save_plots(output_dir, y_true, y_pred, probabilities, metrics)
    metrics["artifacts"] = {
        **report_paths,
        "plots": plot_paths,
    }
    write_json(output_dir / "metrics.json", metrics)

    print("\nMETRICS bidirectional_average_argmax")
    print(f"accuracy={metrics['accuracy']:.6f}")
    for label in LABELS:
        print(f"f1_{label}={metrics['per_class'][label]['f1']:.6f}")
    print(f"macro_f1={metrics['macro_f1']:.6f}")

    print("\nconfusion_matrix rows=true cols=pred labels=" + ",".join(LABELS))
    matrix = np.array(metrics["confusion_matrix"]["rows_true_cols_pred"], dtype=int)
    for label, row in zip(LABELS, matrix, strict=True):
        print(label + ": " + " ".join(str(int(value)) for value in row))

    print("\nroc_auc_ovr_per_class")
    for label in LABELS:
        auc = metrics["per_class"][label]["roc_auc_ovr"]
        auc_text = "nan" if auc is None else f"{auc:.6f}"
        print(f"roc_auc_{label}={auc_text}")
    if metrics["roc_auc_macro_ovr"] is not None:
        print(f"roc_auc_macro_ovr={metrics['roc_auc_macro_ovr']:.6f}")

    print("\nclassification_counts")
    print("pred=" + str(metrics["prediction_counts"]))

    print("\nfirst_errors limit=20")
    errors = 0
    for row, true_label, pred_label, scores in zip(rows, y_true, y_pred, probabilities, strict=True):
        if true_label == pred_label:
            continue
        errors += 1
        if errors <= 20:
            score_text = ", ".join(
                f"{label}:{scores[index]:.3f}"
                for index, label in enumerate(LABELS)
            )
            print(
                f"{row['id']} true={true_label} pred={pred_label} "
                f"scores=[{score_text}]"
            )
    print(f"total_errors={errors}")
    print("\nsaved_artifacts")
    for name, path in report_paths.items():
        print(f"{name}={path}")
    for name, path in plot_paths.items():
        print(f"plot_{name}={path}")


if __name__ == "__main__":
    main()
