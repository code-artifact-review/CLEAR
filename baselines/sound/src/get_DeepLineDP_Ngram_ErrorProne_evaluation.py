from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

DATASETS = (
    # "glance_dataset",
    "linedp_dataset",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEEPLINEDP_PREDICTION_ROOT = (
    PROJECT_ROOT
    / "baselines"
    / "sound"
    / "output"
    / "prediction"
    / "DeepLineDP"
    / "within-release"
)

ERRORPRONE_RESULT_ROOT = (
    PROJECT_ROOT
    / "baselines"
    / "sound"
    / "src"
    / "models"
    / "ErrorProne"
    / "ErrorProne_result"
)

NGRAM_RESULT_ROOT = (
    PROJECT_ROOT
    / "baselines"
    / "sound"
    / "src"
    / "models"
    / "Ngram"
    / "n_gram_result"
)

METRIC_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "clear"
    / "exp"
    / "metrics"
)

DEEPLINEDP_OUTPUT_FILENAME = "DeepLineDP.csv"
ERRORPRONE_OUTPUT_FILENAME = "ErrorProne.csv"
NGRAM_OUTPUT_FILENAME = "Ngram.csv"

DEEPLINEDP_COLUMNS = [
    "project",
    "train",
    "test",
    "filename",
    "file-level-ground-truth",
    "prediction-prob",
    "prediction-label",
    "line-number",
    "line-level-ground-truth",
    "is-comment-line",
    "token",
    "token-attention-score",
    "line-attention-score",
]

ERRORPRONE_COLUMNS = [
    "filename",
    "test-release",
    "line_number",
    "EP_prediction_result",
]

NGRAM_COLUMNS = [
    "train-release",
    "test-release",
    "file-name",
    "line-number",
    "token",
    "token-score",
    "line-score",
]

METRIC_COLUMNS = [
    "release",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "far",
    "d2h",
    "mcc",
    "f1",
    "auc",
    "ifa",
    "recall_0",
    "recall_10",
    "recall_20",
    "recall_30",
    "recall_40",
    "recall_50",
    "recall_60",
    "recall_70",
    "recall_80",
    "recall_90",
    "recall_100",
    "effort@20%recall",
]


def normalize_filename_series(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.replace(r"/+", "/", regex=True)
        .str.replace(r"^\./", "", regex=True)
    )


def parse_binary_series(
        values: pd.Series,
        context: str,
) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    valid_values = {"true", "false", "1", "0"}
    invalid_mask = normalized.isna() | ~normalized.isin(
        valid_values
    )

    if bool(invalid_mask.any()):
        invalid_values = (
            normalized.loc[invalid_mask]
            .drop_duplicates()
            .astype(str)
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Invalid binary values in {context}: {invalid_values}"
        )

    return normalized.isin({"true", "1"}).astype(np.int8)


def parse_numeric_series(
        values: pd.Series,
        context: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_mask = numeric.isna() | ~np.isfinite(numeric)

    if bool(invalid_mask.any()):
        raise ValueError(f"Invalid numeric values in {context}.")

    return numeric.astype(np.float64)


def parse_line_number_series(
        values: pd.Series,
        context: str,
        allow_invalid: bool,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    tolerance = math.sqrt(np.finfo(float).eps)

    valid_mask = (
            numeric.notna()
            & np.isfinite(numeric)
            & (numeric >= 1)
            & np.isclose(
        numeric,
        np.round(numeric),
        rtol=0.0,
        atol=tolerance,
    )
    )

    if not allow_invalid and bool((~valid_mask).any()):
        invalid_values = (
            values.loc[~valid_mask]
            .astype("string")
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"Invalid line numbers in {context}: {invalid_values}"
        )

    result = pd.Series(
        pd.array([pd.NA] * len(values), dtype="Int64"),
        index=values.index,
    )
    result.loc[valid_mask] = np.round(
        numeric.loc[valid_mask]
    ).astype(np.int64)

    return result


def read_csv_header(file_path: Path) -> List[str]:
    return pd.read_csv(
        file_path,
        nrows=0,
    ).columns.astype(str).tolist()


def validate_required_columns(
        file_path: Path,
        actual_columns: Sequence[str],
        required_columns: Sequence[str],
        method_name: str,
) -> None:
    missing_columns = sorted(
        set(required_columns) - set(actual_columns)
    )

    if missing_columns:
        raise ValueError(
            f"{method_name} file is missing required columns: "
            f"{file_path}; missing columns: {missing_columns}"
        )


def discover_deeplinedp_files(
        prediction_dir: Path,
) -> Dict[str, Path]:
    prediction_files = sorted(prediction_dir.glob("*.csv"))

    if not prediction_files:
        raise FileNotFoundError(
            f"No DeepLineDP prediction CSV files were found in: "
            f"{prediction_dir}"
        )

    release_files: Dict[str, Path] = {}

    for prediction_file in prediction_files:
        actual_columns = read_csv_header(prediction_file)
        validate_required_columns(
            file_path=prediction_file,
            actual_columns=actual_columns,
            required_columns=DEEPLINEDP_COLUMNS,
            method_name="DeepLineDP",
        )

        test_values = pd.read_csv(
            prediction_file,
            usecols=["test"],
            dtype={"test": "string"},
            low_memory=False,
        )["test"]

        releases = (
            test_values.dropna()
            .astype("string")
            .str.strip()
        )
        releases = sorted(
            release
            for release in releases.unique().tolist()
            if release
        )

        if len(releases) != 1:
            raise ValueError(
                "Each DeepLineDP CSV must contain exactly one test "
                f"release: {prediction_file}; found: {releases}"
            )

        release = releases[0]

        if release in release_files:
            raise ValueError(
                "Multiple DeepLineDP CSV files were found for release "
                f"{release}: {release_files[release]} and "
                f"{prediction_file}"
            )

        release_files[release] = prediction_file

    return release_files


def load_deeplinedp_line_table(
        prediction_file: Path,
        release: str,
) -> pd.DataFrame:
    use_columns = [
        "test",
        "filename",
        "prediction-prob",
        "prediction-label",
        "line-number",
        "line-level-ground-truth",
        "is-comment-line",
        "line-attention-score",
    ]

    data = pd.read_csv(
        prediction_file,
        usecols=use_columns,
        low_memory=False,
    )

    release_mask = (
            data["test"].astype("string").str.strip() == release
    )
    data = data.loc[release_mask].copy()

    if data.empty:
        raise ValueError(
            f"No DeepLineDP rows were found for release {release}."
        )

    data["filename"] = normalize_filename_series(
        data["filename"]
    )
    data["line-number"] = parse_line_number_series(
        data["line-number"],
        context=f"DeepLineDP predictions for {release}",
        allow_invalid=False,
    )
    data["prediction-prob"] = parse_numeric_series(
        data["prediction-prob"],
        context=f"DeepLineDP file probabilities for {release}",
    )
    data["prediction-label"] = parse_binary_series(
        data["prediction-label"],
        context=f"DeepLineDP file labels for {release}",
    )
    data["line-level-ground-truth"] = parse_binary_series(
        data["line-level-ground-truth"],
        context=f"DeepLineDP line ground truth for {release}",
    )
    data["is-comment-line"] = parse_binary_series(
        data["is-comment-line"],
        context=f"DeepLineDP comment labels for {release}",
    )
    data["line-attention-score"] = parse_numeric_series(
        data["line-attention-score"],
        context=f"DeepLineDP line attention scores for {release}",
    )

    line_rows = data[
        [
            "filename",
            "line-number",
            "prediction-prob",
            "prediction-label",
            "line-level-ground-truth",
            "is-comment-line",
            "line-attention-score",
        ]
    ].drop_duplicates()

    duplicate_line_keys = line_rows.duplicated(
        subset=["filename", "line-number"],
        keep=False,
    )

    if bool(duplicate_line_keys.any()):
        raise ValueError(
            "DeepLineDP contains inconsistent repeated records for "
            f"the same source line in release {release}."
        )

    line_rows = line_rows.loc[
        line_rows["is-comment-line"] == 0
        ].copy()

    if line_rows.empty:
        raise ValueError(
            f"No non-comment lines were found for release {release}."
        )

    file_consistency = (
        line_rows.groupby("filename", sort=False)
        .agg(
            label_count=(
                "prediction-label",
                "nunique",
            ),
            probability_min=(
                "prediction-prob",
                "min",
            ),
            probability_max=(
                "prediction-prob",
                "max",
            ),
        )
        .reset_index()
    )

    inconsistent_files = (
            (file_consistency["label_count"] != 1)
            | ~np.isclose(
        file_consistency["probability_min"],
        file_consistency["probability_max"],
        rtol=0.0,
        atol=1e-12,
    )
    )

    if bool(inconsistent_files.any()):
        raise ValueError(
            "DeepLineDP contains inconsistent file-level predictions "
            f"within release {release}."
        )

    line_rows = line_rows.rename(
        columns={
            "line-number": "line_number",
            "prediction-prob": "file_prediction_probability",
            "prediction-label": "predicted_file_bug",
            "line-level-ground-truth": "line_label",
            "line-attention-score": "deeplinedp_line_score",
        }
    )

    return line_rows[
        [
            "filename",
            "line_number",
            "line_label",
            "predicted_file_bug",
            "file_prediction_probability",
            "deeplinedp_line_score",
        ]
    ].reset_index(drop=True)


def load_errorprone_line_scores(
        result_file: Path,
        release: str,
) -> pd.DataFrame:
    if not result_file.exists():
        raise FileNotFoundError(
            f"ErrorProne result file does not exist: {result_file}"
        )

    actual_columns = read_csv_header(result_file)
    validate_required_columns(
        file_path=result_file,
        actual_columns=actual_columns,
        required_columns=ERRORPRONE_COLUMNS,
        method_name="ErrorProne",
    )

    data = pd.read_csv(
        result_file,
        usecols=ERRORPRONE_COLUMNS,
        low_memory=False,
    )

    release_mask = (
            data["test-release"].astype("string").str.strip()
            == release
    )
    data = data.loc[release_mask].copy()

    raw_line_numbers = parse_line_number_series(
        data["line_number"],
        context=f"ErrorProne results for {release}",
        allow_invalid=True,
    )

    raw_scores = (
        data["EP_prediction_result"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    valid_score_mask = raw_scores.isin(
        {"true", "false", "1", "0"}
    )

    normalized = pd.DataFrame(
        {
            "filename": normalize_filename_series(
                data["filename"]
            ),
            "line_number": raw_line_numbers,
            "line_score": raw_scores.isin(
                {"true", "1"}
            ).astype(np.float64),
            "valid_score": valid_score_mask,
        }
    )

    valid_mask = (
            normalized["filename"].notna()
            & (normalized["filename"] != "")
            & normalized["line_number"].notna()
            & normalized["valid_score"]
    )

    normalized = normalized.loc[
        valid_mask,
        ["filename", "line_number", "line_score"],
    ].copy()
    normalized["line_number"] = normalized[
        "line_number"
    ].astype(np.int64)

    return (
        normalized.groupby(
            ["filename", "line_number"],
            as_index=False,
            sort=False,
        )["line_score"]
        .max()
    )


def load_ngram_line_scores(
        result_file: Path,
        release: str,
) -> pd.DataFrame:
    if not result_file.exists():
        raise FileNotFoundError(
            f"Ngram result file does not exist: {result_file}"
        )

    actual_columns = read_csv_header(result_file)
    validate_required_columns(
        file_path=result_file,
        actual_columns=actual_columns,
        required_columns=NGRAM_COLUMNS,
        method_name="Ngram",
    )

    ngram_use_columns = [
        "test-release",
        "file-name",
        "line-number",
        "line-score",
    ]

    try:
        data = pd.read_csv(
            result_file,
            usecols=ngram_use_columns,
            low_memory=False,
        )
    except pd.errors.ParserError:
        data = pd.read_csv(
            result_file,
            usecols=ngram_use_columns,
            engine="python",
            quoting=csv.QUOTE_NONE,
        )

    release_mask = (
            data["test-release"].astype("string").str.strip()
            == release
    )
    data = data.loc[release_mask].copy()

    line_numbers = parse_line_number_series(
        data["line-number"],
        context=f"Ngram results for {release}",
        allow_invalid=True,
    )
    line_scores = pd.to_numeric(
        data["line-score"],
        errors="coerce",
    )

    normalized = pd.DataFrame(
        {
            "filename": normalize_filename_series(
                data["file-name"]
            ),
            "line_number": line_numbers,
            "line_score": line_scores,
        }
    )

    valid_mask = (
            normalized["filename"].notna()
            & (normalized["filename"] != "")
            & normalized["line_number"].notna()
            & normalized["line_score"].notna()
            & np.isfinite(normalized["line_score"])
    )

    normalized = normalized.loc[
        valid_mask
    ].copy()
    normalized["line_number"] = normalized[
        "line_number"
    ].astype(np.int64)

    return (
        normalized.groupby(
            ["filename", "line_number"],
            as_index=False,
            sort=False,
        )["line_score"]
        .max()
    )


def create_global_ranking(
        line_table: pd.DataFrame,
        line_scores: pd.DataFrame,
        score_column: str,
) -> pd.DataFrame:
    if score_column in line_table.columns:
        ranked = line_table.rename(
            columns={score_column: "line_score"}
        ).copy()
    else:
        ranked = line_table.merge(
            line_scores,
            on=["filename", "line_number"],
            how="left",
            validate="one_to_one",
        )
        ranked["line_score"] = ranked[
            "line_score"
        ].fillna(0.0)

    ranked = ranked.sort_values(
        by=[
            "predicted_file_bug",
            "file_prediction_probability",
            "line_score",
            "filename",
            "line_number",
        ],
        ascending=[
            False,
            False,
            False,
            True,
            True,
        ],
        kind="mergesort",
        ignore_index=True,
    )

    if len(ranked) != len(line_table):
        raise ValueError(
            "The ranking does not contain all non-comment test lines."
        )

    if bool(
            ranked.duplicated(
                subset=["filename", "line_number"]
            ).any()
    ):
        raise ValueError(
            "The ranking contains duplicate source lines."
        )

    return ranked


def safe_divide(
        numerator: float,
        denominator: float,
) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def calculate_rank_auc(labels: np.ndarray) -> float:
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)

    if positive_count == 0 or negative_count == 0:
        return float("nan")

    descending_ranks = (
            len(labels)
            - np.arange(
        len(labels),
        dtype=np.float64,
    )
    )
    positive_rank_sum = float(
        descending_ranks[labels == 1].sum()
    )

    return float(
        (
                positive_rank_sum
                - positive_count
                * (positive_count + 1)
                / 2.0
        )
        / (
                positive_count
                * negative_count
        )
    )


def calculate_global_metrics(
        ranked_lines: pd.DataFrame,
        release: str,
) -> Dict[str, float | int | str]:
    labels = ranked_lines["line_label"].to_numpy(
        dtype=np.int64,
        copy=False,
    )
    total_lines = int(len(labels))

    if total_lines == 0:
        raise ValueError(
            f"No evaluable lines were found for release {release}."
        )

    total_defective_lines = int(labels.sum())
    cumulative_defects = np.cumsum(labels)

    inspected_20_count = min(
        total_lines,
        max(
            1,
            int(
                math.floor(
                    total_lines * 0.2
                )
            ),
        ),
    )

    tp = int(
        cumulative_defects[
            inspected_20_count - 1
            ]
    )
    fp = int(inspected_20_count - tp)
    fn = int(total_defective_lines - tp)
    tn = int(
        total_lines
        - tp
        - fp
        - fn
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )
    recall = safe_divide(
        tp,
        tp + fn,
    )
    far = safe_divide(
        fp,
        fp + tn,
    )
    d2h = (
            math.sqrt(
                (1.0 - recall) ** 2
                + far ** 2
            )
            / math.sqrt(2.0)
    )

    mcc_denominator = (
            float(tp + fp)
            * float(tp + fn)
            * float(tn + fp)
            * float(tn + fn)
    )

    if mcc_denominator <= 0.0:
        mcc = 0.0
    else:
        mcc = float(
            (
                    float(tp) * float(tn)
                    - float(fp) * float(fn)
            )
            / math.sqrt(mcc_denominator)
        )

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(
            2.0
            * precision
            * recall
            / (
                    precision
                    + recall
            )
        )

    positive_positions = np.flatnonzero(
        labels == 1
    )
    ifa = (
        int(positive_positions[0])
        if len(positive_positions) > 0
        else total_lines
    )

    recall_metrics: Dict[str, float] = {}

    for percentage in range(0, 101, 10):
        if percentage == 0:
            found_defects = 0
        else:
            inspected_count = int(
                math.floor(
                    total_lines
                    * percentage
                    / 100.0
                )
            )

            if percentage == 100:
                inspected_count = total_lines

            inspected_count = min(
                total_lines,
                max(
                    1,
                    inspected_count,
                ),
            )
            found_defects = int(
                cumulative_defects[
                    inspected_count - 1
                    ]
            )

        recall_metrics[
            f"recall_{percentage}"
        ] = safe_divide(
            found_defects,
            total_defective_lines,
        )

    if total_defective_lines == 0:
        effort_at_20_recall = 0.0
    else:
        required_defects = max(
            1,
            int(
                math.ceil(
                    total_defective_lines
                    * 0.2
                )
            ),
        )
        target_positions = np.flatnonzero(
            cumulative_defects
            >= required_defects
        )
        effort_at_20_recall = (
            1.0
            if len(target_positions) == 0
            else float(
                (
                        int(target_positions[0])
                        + 1
                )
                / total_lines
            )
        )

    result: Dict[
        str,
        float | int | str,
    ] = {
        "release": release,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "far": far,
        "d2h": d2h,
        "mcc": mcc,
        "f1": f1,
        "auc": calculate_rank_auc(
            labels
        ),
        "ifa": ifa,
        **recall_metrics,
        "effort@20%recall": (
            effort_at_20_recall
        ),
    }

    return result


def evaluate_release(
        release: str,
        prediction_file: Path,
        errorprone_result_dir: Path,
        ngram_result_dir: Path,
) -> Mapping[
    str,
    Dict[str, float | int | str],
]:
    line_table = load_deeplinedp_line_table(
        prediction_file=prediction_file,
        release=release,
    )

    deeplinedp_ranking = create_global_ranking(
        line_table=line_table,
        line_scores=pd.DataFrame(),
        score_column="deeplinedp_line_score",
    )

    errorprone_file = (
            errorprone_result_dir
            / f"{release}-line-lvl-result.txt"
    )
    errorprone_scores = load_errorprone_line_scores(
        result_file=errorprone_file,
        release=release,
    )
    errorprone_ranking = create_global_ranking(
        line_table=line_table.drop(
            columns=["deeplinedp_line_score"]
        ),
        line_scores=errorprone_scores,
        score_column="errorprone_line_score",
    )

    ngram_file = (
            ngram_result_dir
            / f"{release}-line-lvl-result.txt"
    )
    ngram_scores = load_ngram_line_scores(
        result_file=ngram_file,
        release=release,
    )
    ngram_ranking = create_global_ranking(
        line_table=line_table.drop(
            columns=["deeplinedp_line_score"]
        ),
        line_scores=ngram_scores,
        score_column="ngram_line_score",
    )

    return {
        "DeepLineDP": calculate_global_metrics(
            ranked_lines=deeplinedp_ranking,
            release=release,
        ),
        "ErrorProne": calculate_global_metrics(
            ranked_lines=errorprone_ranking,
            release=release,
        ),
        "Ngram": calculate_global_metrics(
            ranked_lines=ngram_ranking,
            release=release,
        ),
    }


def write_metric_results(
        records: Sequence[Mapping[str, object]],
        output_file: Path,
) -> None:
    result = pd.DataFrame.from_records(
        records
    )

    if result.empty:
        result = pd.DataFrame(
            columns=METRIC_COLUMNS
        )
    else:
        result = (
            result.loc[
            :,
            METRIC_COLUMNS,
            ]
            .sort_values(
                by="release",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    result.to_csv(
        output_file,
        index=False,
    )


def main() -> None:
    for dataset_name in DATASETS:
        prediction_dir = (
                DEEPLINEDP_PREDICTION_ROOT
                / dataset_name
                / "0"
        )
        errorprone_result_dir = (
                ERRORPRONE_RESULT_ROOT
                / dataset_name
        )
        ngram_result_dir = (
                NGRAM_RESULT_ROOT
                / dataset_name
        )
        metric_output_dir = (
                METRIC_OUTPUT_ROOT
                / dataset_name
        )
        metric_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        deeplinedp_output_file = (
                metric_output_dir
                / DEEPLINEDP_OUTPUT_FILENAME
        )
        errorprone_output_file = (
                metric_output_dir
                / ERRORPRONE_OUTPUT_FILENAME
        )
        ngram_output_file = (
                metric_output_dir
                / NGRAM_OUTPUT_FILENAME
        )

        print(
            f"========== Evaluating {dataset_name} =========="
        )

        release_files = discover_deeplinedp_files(
            prediction_dir=prediction_dir
        )

        deeplinedp_records: List[
            Mapping[str, object]
        ] = []
        errorprone_records: List[
            Mapping[str, object]
        ] = []
        ngram_records: List[
            Mapping[str, object]
        ] = []

        for release in sorted(release_files):
            print(release)
            release_results = evaluate_release(
                release=release,
                prediction_file=release_files[release],
                errorprone_result_dir=errorprone_result_dir,
                ngram_result_dir=ngram_result_dir,
            )

            deeplinedp_records.append(
                release_results["DeepLineDP"]
            )
            errorprone_records.append(
                release_results["ErrorProne"]
            )
            ngram_records.append(
                release_results["Ngram"]
            )

        write_metric_results(
            records=deeplinedp_records,
            output_file=deeplinedp_output_file,
        )
        write_metric_results(
            records=errorprone_records,
            output_file=errorprone_output_file,
        )
        write_metric_results(
            records=ngram_records,
            output_file=ngram_output_file,
        )

        print(
            f"[{dataset_name}] Saved DeepLineDP metrics to "
            f"{deeplinedp_output_file}"
        )
        print(
            f"[{dataset_name}] Saved ErrorProne metrics to "
            f"{errorprone_output_file}"
        )
        print(
            f"[{dataset_name}] Saved Ngram metrics to "
            f"{ngram_output_file}"
        )

    print("done")


if __name__ == "__main__":
    main()
