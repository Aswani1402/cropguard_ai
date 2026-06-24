from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

CUSTOM_RESULTS_PATH = METRICS_DIR / "custom_cnn_results.csv"
CUSTOM_REPORT_PATH = METRICS_DIR / "custom_cnn_classification_report.csv"

EFFICIENT_RESULTS_PATH = METRICS_DIR / "efficientnetv2b0_results.csv"
EFFICIENT_REPORT_PATH = METRICS_DIR / "efficientnetv2b0_classification_report.csv"

MOBILENET_RESULTS_PATH = METRICS_DIR / "mobilenetv3large_results.csv"
MOBILENET_REPORT_PATH = METRICS_DIR / "mobilenetv3large_classification_report.csv"

OUTPUT_CSV = METRICS_DIR / "model_comparison.csv"
OUTPUT_MD = METRICS_DIR / "model_comparison.md"
OUTPUT_CHART = METRICS_DIR / "model_comparison_accuracy.png"


def check_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def load_result_metrics(results_path: Path):
    df = pd.read_csv(results_path)

    if df.empty:
        raise ValueError(f"Empty results file: {results_path}")

    row = df.iloc[0].to_dict()

    return {
        "accuracy": float(row["accuracy"]),
        "loss": float(row["loss"]),
        "top_3_accuracy": float(row["top_3_accuracy"]),
        "train_images": int(row["train_images"]),
        "valid_images": int(row["valid_images"]),
        "num_classes": int(row["num_classes"]),
    }


def load_f1_scores(report_path: Path):
    df = pd.read_csv(report_path)

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "label"})
    df = df.set_index("label")

    return {
        "macro_f1": float(df.loc["macro avg", "f1-score"]),
        "weighted_f1": float(df.loc["weighted avg", "f1-score"]),
    }


def build_row(model_name, role, results_path, report_path):
    metrics = load_result_metrics(results_path)
    f1_scores = load_f1_scores(report_path)

    return {
        "model": model_name,
        "role": role,
        "accuracy": metrics["accuracy"],
        "macro_f1": f1_scores["macro_f1"],
        "weighted_f1": f1_scores["weighted_f1"],
        "top_3_accuracy": metrics["top_3_accuracy"],
        "loss": metrics["loss"],
        "train_images": metrics["train_images"],
        "valid_images": metrics["valid_images"],
        "num_classes": metrics["num_classes"],
    }


def main():
    required_files = [
        CUSTOM_RESULTS_PATH,
        CUSTOM_REPORT_PATH,
        EFFICIENT_RESULTS_PATH,
        EFFICIENT_REPORT_PATH,
        MOBILENET_RESULTS_PATH,
        MOBILENET_REPORT_PATH,
    ]

    for path in required_files:
        check_file(path)

    rows = [
        build_row(
            "Custom CNN Baseline",
            "Baseline model trained from scratch",
            CUSTOM_RESULTS_PATH,
            CUSTOM_REPORT_PATH,
        ),
        build_row(
            "EfficientNetV2B0",
            "Main transfer learning model",
            EFFICIENT_RESULTS_PATH,
            EFFICIENT_REPORT_PATH,
        ),
        build_row(
            "MobileNetV3Large",
            "Lightweight deployment-friendly model",
            MOBILENET_RESULTS_PATH,
            MOBILENET_REPORT_PATH,
        ),
    ]

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("accuracy", ascending=False)

    comparison_df.to_csv(OUTPUT_CSV, index=False)

    markdown_table = comparison_df.copy()
    metric_columns = ["accuracy", "macro_f1", "weighted_f1", "top_3_accuracy", "loss"]

    for col in metric_columns:
        markdown_table[col] = markdown_table[col].map(lambda x: f"{x:.4f}")

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Model Comparison\n\n")
        f.write(markdown_table.to_markdown(index=False))
        f.write("\n")

    plt.figure(figsize=(9, 5))
    plt.bar(comparison_df["model"], comparison_df["accuracy"])
    plt.ylabel("Validation Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0.94, 1.00)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=200)
    plt.close()

    print("\nModel comparison completed.")
    print(comparison_df.to_string(index=False))

    print("\nBest model:")
    print(comparison_df.iloc[0][["model", "accuracy", "macro_f1", "top_3_accuracy"]])

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_MD)
    print(OUTPUT_CHART)


if __name__ == "__main__":
    main()