import ast
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

from clear.src.my_utils.helper import (
    glance_projects,
    linedp_projects,
)

DATA_ROOT = '../Data'
DEFAULT_FILE_MODEL_NAME = 'logistic_regression'
DEFAULT_EFFORT_RATIO = 0.2

CLEAR_LINE_SCORE_METHODS = (
    'sum',
    'max',
    'top2avg',
    'top3avg',
)

RECALL_PERCENTAGES = tuple(range(0, 101, 10))
CLEAR_RESULT_FILENAMES = {
    'max': 'CLEAR-MAX.csv',
    'top3avg': 'CLEAR-TOP3AVG.csv',
    'sum': 'CLEAR-SUM.csv',
    'top2avg': 'CLEAR-TOP2AVG.csv',
}

TOKEN_SCORE_TYPES = (
    'confidence_enhanced',
    'suspiciousness_only',
)

TOKEN_SCORE_TYPES_TO_EVALUATE = (
    'confidence_enhanced',
)

LINE_COUNTERFACTUAL_RESULT_DIRECTORIES = {
    'confidence_enhanced': 'line_counterfactual_analysis',
    'suspiciousness_only': (
        'line_counterfactual_analysis_without_confidence'
    ),
}

RANKING_EVALUATION_RESULT_DIRECTORIES = {
    'confidence_enhanced': (
        'ranking_evaluation_without_file_prediction_'
        'and_file_counterfactual'
    ),
    'suspiciousness_only': (
        'ranking_evaluation_without_confidence_'
        'without_file_prediction_and_file_counterfactual'
    ),
}

TOKEN_SCORE_FILENAMES = {
    'confidence_enhanced': (
        '{train_release}_confidence_enhanced_token_scores.json'
    ),
    'suspiciousness_only': (
        '{train_release}_suspiciousness_token_scores.json'
    ),
}

TOKEN_VECTORIZER = CountVectorizer(lowercase=False)
TOKENIZER = TOKEN_VECTORIZER.build_tokenizer()


@dataclass(frozen=True)
class RankingStrategy:

    name: str
    sort_columns: Tuple[str, ...]
    prefix_mode: Optional[str] = None
    prefix_multiplier: float = 1.0
    fixed_prefix_ratio: Optional[float] = None


def build_ranking_strategies() -> Tuple[RankingStrategy, ...]:


    return (
        RankingStrategy(
            name=(
                'global_line_causal_then_suspiciousness'
            ),
            sort_columns=(
                'contains_line_counterfactual_token',
                'suspicion_score',
            ),
        ),
    )


RANKING_STRATEGIES = build_ranking_strategies()


def safe_divide(numerator: float, denominator: float) -> float:


    return 0.0 if denominator == 0 else numerator / denominator


def parse_token_list(value) -> List[str]:


    if isinstance(value, list):
        return [str(token) for token in value]

    if value is None or (
            isinstance(value, float)
            and math.isnan(value)
    ):
        return []

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            if isinstance(parsed, list):
                return [str(token) for token in parsed]
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue

    return []


def load_json_token_set(path: str) -> set:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Counterfactual token file not found: {path}'
        )

    with open(path, 'r', encoding='utf-8') as input_file:
        tokens = json.load(input_file)

    if not isinstance(tokens, list):
        raise ValueError(
            f'Expected a JSON token list in {path}.'
        )

    return {str(token) for token in tokens}


def validate_token_score_type(
        token_score_type: str,
) -> None:

    if token_score_type not in TOKEN_SCORE_TYPES:
        raise ValueError(
            'Unsupported token score type: '
            f'{token_score_type}. Supported values: '
            f'{TOKEN_SCORE_TYPES}'
        )


def get_line_counterfactual_result_directory(
        token_score_type: str,
) -> str:

    validate_token_score_type(token_score_type)
    return LINE_COUNTERFACTUAL_RESULT_DIRECTORIES[
        token_score_type
    ]


def get_ranking_evaluation_result_directory(
        token_score_type: str,
) -> str:

    validate_token_score_type(token_score_type)
    return RANKING_EVALUATION_RESULT_DIRECTORIES[
        token_score_type
    ]


def get_line_prediction_path(
        dataset_name: str,
        test_release: str,
        aggregation_method: str,
        model_name: str,
        token_score_type: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        get_line_counterfactual_result_directory(
            token_score_type
        ),
        model_name,
        'predictions',
        f'{test_release}_clear_{aggregation_method}.csv',
    )


def get_line_counterfactual_token_path(
        dataset_name: str,
        train_release: str,
        aggregation_method: str,
        model_name: str,
        token_score_type: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        get_line_counterfactual_result_directory(
            token_score_type
        ),
        model_name,
        'tokens',
        f'{train_release}_clear_{aggregation_method}_'
        'counterfactual_tokens.json',
    )


def get_token_score_path(
        dataset_name: str,
        train_release: str,
        token_score_type: str,
) -> str:

    validate_token_score_type(token_score_type)

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'token_scores',
        TOKEN_SCORE_FILENAMES[token_score_type].format(
            train_release=train_release
        ),
    )


def get_preprocessed_line_path(
        dataset_name: str,
        release: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'preprocessed_data',
        f'{release}.csv',
    )


def load_preprocessed_line_data(
        dataset_name: str,
        release: str,
) -> pd.DataFrame:

    input_path = get_preprocessed_line_path(
        dataset_name,
        release,
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'Preprocessed line data not found: {input_path}'
        )

    dataframe = pd.read_csv(input_path, encoding='utf-8')

    required_columns = {
        'filename',
        'code_line',
        'line-label',
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required preprocessed columns: '
            f'{sorted(missing_columns)}'
        )

    dataframe = dataframe.copy()
    dataframe['filename'] = dataframe['filename'].astype(str)
    dataframe['code_line'] = (
        dataframe['code_line'].fillna('').astype(str)
    )
    dataframe['line-label'] = pd.to_numeric(
        dataframe['line-label'],
        errors='raise',
    ).astype(int)

    invalid_labels = (
            set(dataframe['line-label'].unique()) - {0, 1}
    )
    if invalid_labels:
        raise ValueError(
            'Line-level labels must be binary values 0 or 1. '
            f'Invalid labels: {sorted(invalid_labels)}'
        )

    return dataframe


def calculate_line_risk_score(
        tokens: Iterable[str],
        token_risk_scores: Dict[str, float],
        aggregation_method: str,
) -> float:

    scores = [
        token_risk_scores.get(token, 0.0)
        for token in tokens
    ]

    if not scores:
        return 0.0

    if aggregation_method == 'sum':
        return float(sum(scores))

    if aggregation_method == 'max':
        return float(max(scores))

    if aggregation_method == 'top2avg':
        top_scores = sorted(scores, reverse=True)[:2]
        return float(sum(top_scores) / len(top_scores))

    if aggregation_method == 'top3avg':
        top_scores = sorted(scores, reverse=True)[:3]
        return float(sum(top_scores) / len(top_scores))

    raise ValueError(
        f'Unsupported aggregation method: {aggregation_method}'
    )


def add_line_identifiers(dataframe: pd.DataFrame) -> pd.DataFrame:

    output_dataframe = dataframe.copy()
    output_dataframe['_original_order'] = np.arange(
        len(output_dataframe),
        dtype=int,
    )

    if 'line_number' in output_dataframe.columns:
        line_numbers = output_dataframe['line_number'].astype(str)
    else:
        line_numbers = (
                output_dataframe.groupby('filename').cumcount() + 1
        ).astype(str)
        output_dataframe['line_number'] = line_numbers

    output_dataframe['line_id'] = (
            output_dataframe['filename'].astype(str)
            + ':'
            + line_numbers
    )

    return output_dataframe


def add_line_counterfactual_features(
        dataframe: pd.DataFrame,
        line_counterfactual_tokens: set,
) -> pd.DataFrame:


    output_dataframe = dataframe.copy()

    line_occurrence_counts: List[int] = []
    line_unique_counts: List[int] = []

    for code_line in output_dataframe['code_line']:
        tokens = TOKENIZER(str(code_line))
        unique_tokens = set(tokens)

        line_occurrence_counts.append(
            sum(
                1
                for token in tokens
                if token in line_counterfactual_tokens
            )
        )
        line_unique_counts.append(
            len(unique_tokens & line_counterfactual_tokens)
        )

    output_dataframe[
        'line_counterfactual_token_count'
    ] = line_occurrence_counts
    output_dataframe[
        'line_counterfactual_unique_token_count'
    ] = line_unique_counts
    output_dataframe[
        'contains_line_counterfactual_token'
    ] = (
            output_dataframe[
                'line_counterfactual_token_count'
            ] > 0
    ).astype(int)

    return output_dataframe


def calculate_training_line_defect_rate(
        dataset_name: str,
        train_release: str,
) -> float:

    historical_dataframe = load_preprocessed_line_data(
        dataset_name,
        train_release,
    )

    if historical_dataframe.empty:
        raise ValueError(
            f'No historical lines found for {train_release}.'
        )

    return float(
        (historical_dataframe['line-label'] == 1).sum()
        / len(historical_dataframe)
    )


def rank_indices_by_columns(
        dataframe: pd.DataFrame,
        sort_columns: Sequence[str],
        stable_order: Optional[np.ndarray] = None,
) -> np.ndarray:

    effective_columns = list(sort_columns)

    missing_columns = (
            set(effective_columns) - set(dataframe.columns)
    )
    if missing_columns:
        raise ValueError(
            'Missing ranking columns: '
            f'{sorted(missing_columns)}'
        )

    if stable_order is None:
        stable_order = dataframe[
            '_original_order'
        ].to_numpy(dtype=int)

    lexicographic_keys: List[np.ndarray] = [
        np.asarray(stable_order, dtype=int)
    ]

    for column in reversed(effective_columns):
        values = pd.to_numeric(
            dataframe[column],
            errors='raise',
        ).to_numpy(dtype=float)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=np.finfo(float).max,
            neginf=np.finfo(float).min,
        )
        lexicographic_keys.append(-values)

    return np.lexsort(tuple(lexicographic_keys))


def resolve_prefix_ratio(
        dataframe: pd.DataFrame,
        strategy: RankingStrategy,
) -> float:

    if strategy.fixed_prefix_ratio is not None:
        return float(
            min(max(strategy.fixed_prefix_ratio, 0.0), 1.0)
        )

    if 'line_defect_rate' not in dataframe.columns:
        raise ValueError(
            'line_defect_rate is required for defect-rate prefix '
            're-ranking.'
        )

    valid_rates = pd.to_numeric(
        dataframe['line_defect_rate'],
        errors='coerce',
    ).dropna()

    if valid_rates.empty:
        raise ValueError(
            'No valid line_defect_rate is available.'
        )

    base_rate = float(valid_rates.iloc[0])
    return float(min(
        max(base_rate * strategy.prefix_multiplier, 0.0),
        1.0,
    ))


def apply_ranking_strategy(
        dataframe: pd.DataFrame,
        strategy: RankingStrategy,
) -> np.ndarray:

    if strategy.prefix_mode is None:
        return rank_indices_by_columns(
            dataframe,
            strategy.sort_columns,
        )

    baseline_order = rank_indices_by_columns(
        dataframe,
        ('suspicion_score',),
    )

    prefix_ratio = resolve_prefix_ratio(
        dataframe,
        strategy,
    )
    prefix_size = int(
        math.ceil(len(dataframe) * prefix_ratio)
    )
    prefix_size = min(
        max(prefix_size, 1),
        len(dataframe),
    )

    prefix_indices = baseline_order[:prefix_size]
    remainder_indices = baseline_order[prefix_size:]

    prefix_dataframe = (
        dataframe.iloc[prefix_indices]
        .copy()
        .reset_index(drop=True)
    )
    prefix_dataframe['_original_order'] = np.arange(
        len(prefix_dataframe),
        dtype=int,
    )

    prefix_local_order = rank_indices_by_columns(
        prefix_dataframe,
        strategy.sort_columns,
    )
    reranked_prefix_indices = prefix_indices[
        prefix_local_order
    ]

    return np.concatenate([
        reranked_prefix_indices,
        remainder_indices,
    ])


def calculate_ranking_metrics(
        dataframe: pd.DataFrame,
        ranking_indices: np.ndarray,
) -> dict:

    labels = dataframe[
        'line-label'
    ].to_numpy(dtype=int)[ranking_indices]

    total_lines = len(labels)
    total_defective_lines = int(labels.sum())

    if total_lines == 0:
        raise ValueError(
            'Cannot evaluate an empty line ranking.'
        )

    cumulative_defective_lines = np.cumsum(labels)

    effort_20_count = max(
        1,
        int(total_lines * DEFAULT_EFFORT_RATIO),
    )
    effort_20_count = min(
        effort_20_count,
        total_lines,
    )

    tp = int(
        cumulative_defective_lines[
            effort_20_count - 1
            ]
    )
    fp = int(effort_20_count - tp)
    fn = int(total_defective_lines - tp)
    tn = int(total_lines - tp - fp - fn)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    far = safe_divide(fp, fp + tn)

    d2h = math.sqrt(
        (1.0 - recall) ** 2 + far ** 2
    ) / math.sqrt(2.0)

    mcc_denominator = (
            (tp + fp)
            * (tp + fn)
            * (tn + fp)
            * (tn + fn)
    )
    mcc = (
        0.0
        if mcc_denominator == 0
        else (
                (tp * tn - fp * fn)
                / math.sqrt(mcc_denominator)
        )
    )

    f1 = (
        0.0
        if precision + recall == 0
        else (
                2.0
                * precision
                * recall
                / (precision + recall)
        )
    )

    positive_positions = np.flatnonzero(labels == 1)
    ifa = (
        int(positive_positions[0])
        if len(positive_positions) > 0
        else total_lines
    )

    recall_metrics = {}
    for percentage in RECALL_PERCENTAGES:
        if percentage == 0:
            inspected_count = 0
            found_defective_lines = 0
        else:
            inspected_count = int(
                total_lines * percentage / 100
            )
            if percentage == 100:
                inspected_count = total_lines
            inspected_count = min(
                max(inspected_count, 1),
                total_lines,
            )
            found_defective_lines = int(
                cumulative_defective_lines[
                    inspected_count - 1
                    ]
            )

        recall_metrics[
            f'recall_{percentage}'
        ] = safe_divide(
            found_defective_lines,
            total_defective_lines,
        )

    if total_defective_lines == 0:
        effort_at_20_recall = 0.0
    else:
        required_defective_lines = max(
            1,
            int(math.ceil(
                total_defective_lines * 0.2
            )),
        )
        target_positions = np.flatnonzero(
            cumulative_defective_lines
            >= required_defective_lines
        )
        if len(target_positions) == 0:
            effort_at_20_recall = 1.0
        else:
            effort_at_20_recall = float(
                (target_positions[0] + 1)
                / total_lines
            )

    if len(np.unique(labels)) == 2:
        rank_scores = np.linspace(
            1.0,
            0.0,
            num=total_lines,
            endpoint=False,
        )
        auc = float(
            roc_auc_score(labels, rank_scores)
        )
    else:
        auc = float('nan')

    metrics = {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'far': far,
        'd2h': d2h,
        'mcc': mcc,
        'f1': f1,
        'auc': auc,
        'ifa': ifa,
        **recall_metrics,
        'effort@20%recall': effort_at_20_recall,
        'total_lines': total_lines,
        'total_defective_lines': total_defective_lines,
    }

    return metrics


def prepare_test_dataframe(
        dataset_name: str,
        train_release: str,
        test_release: str,
        aggregation_method: str,
        model_name: str,
        token_score_type: str,
) -> pd.DataFrame:


    line_counterfactual_tokens = load_json_token_set(
        get_line_counterfactual_token_path(
            dataset_name=dataset_name,
            train_release=train_release,
            aggregation_method=aggregation_method,
            model_name=model_name,
            token_score_type=token_score_type,
        )
    )
    training_line_defect_rate = (
        calculate_training_line_defect_rate(
            dataset_name,
            train_release,
        )
    )

    line_prediction_path = get_line_prediction_path(
        dataset_name=dataset_name,
        test_release=test_release,
        aggregation_method=aggregation_method,
        model_name=model_name,
        token_score_type=token_score_type,
    )
    if not os.path.exists(line_prediction_path):
        raise FileNotFoundError(
            f'Line prediction file not found: {line_prediction_path}'
        )

    dataframe = pd.read_csv(
        line_prediction_path,
        encoding='utf-8',
    )
    score_column = f'clear_{aggregation_method}'

    required_columns = {
        'filename',
        'code_line',
        'line-label',
        score_column,
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required target line columns: '
            f'{sorted(missing_columns)}'
        )

    dataframe = dataframe.copy()
    dataframe['filename'] = dataframe['filename'].astype(str)
    dataframe['code_line'] = (
        dataframe['code_line'].fillna('').astype(str)
    )
    dataframe['line-label'] = pd.to_numeric(
        dataframe['line-label'],
        errors='raise',
    ).astype(int)
    dataframe['suspicion_score'] = pd.to_numeric(
        dataframe[score_column],
        errors='raise',
    )

    file_level_columns = [
        column
        for column in (
            'file_pred_prob',
            'predicted_file_bug',
        )
        if column in dataframe.columns
    ]
    if file_level_columns:
        dataframe = dataframe.drop(
            columns=file_level_columns
        )

    dataframe = add_line_counterfactual_features(
        dataframe,
        line_counterfactual_tokens,
    )
    dataframe = add_line_identifiers(dataframe)
    dataframe['line_defect_rate'] = (
        training_line_defect_rate
    )
    dataframe['aggregation_method'] = aggregation_method
    dataframe['token_score_type'] = token_score_type

    return dataframe


def build_evaluation_row(
        train_release: str,
        test_release: str,
        aggregation_method: str,
        strategy_name: str,
        token_score_type: str,
        metrics: dict,
) -> dict:

    return {
        'train_release': train_release,
        'test_release': test_release,
        'aggregation_method': aggregation_method,
        'ranking_strategy': strategy_name,
        'token_score_type': token_score_type,
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'tn': metrics['tn'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'far': metrics['far'],
        'd2h': metrics['d2h'],
        'mcc': metrics['mcc'],
        'f1': metrics['f1'],
        'auc': metrics['auc'],
        'ifa': metrics['ifa'],
        **{
            f'recall_{percentage}': (
                metrics[f'recall_{percentage}']
            )
            for percentage in RECALL_PERCENTAGES
        },
        'effort@20%recall': (
            metrics['effort@20%recall']
        ),
    }


def save_clear_method_metric_csv(
        dataset_name: str,
        aggregation_method: str,
        evaluation_dataframe: pd.DataFrame,
        model_name: str,
        token_score_type: str,
) -> str:


    if aggregation_method not in CLEAR_RESULT_FILENAMES:
        raise ValueError(
            'Unsupported CLEAR aggregation method: '
            f'{aggregation_method}'
        )

    required_columns = {
        'test_release',
        'tp',
        'fp',
        'fn',
        'tn',
        'precision',
        'recall',
        'far',
        'd2h',
        'mcc',
        'f1',
        'auc',
        'ifa',
        *{
            f'recall_{percentage}'
            for percentage in RECALL_PERCENTAGES
        },
        'effort@20%recall',
    }
    missing_columns = (
            required_columns - set(evaluation_dataframe.columns)
    )
    if missing_columns:
        raise ValueError(
            'Missing required CLEAR metric columns: '
            f'{sorted(missing_columns)}'
        )

    clear_metric_columns = [
        'release',
        'tp',
        'fp',
        'fn',
        'tn',
        'precision',
        'recall',
        'far',
        'd2h',
        'mcc',
        'f1',
        'auc',
        'ifa',
        *[
            f'recall_{percentage}'
            for percentage in RECALL_PERCENTAGES
        ],
        'effort@20%recall',
    ]

    clear_metric_dataframe = (
        evaluation_dataframe.rename(
            columns={'test_release': 'release'}
        )
        .reindex(columns=clear_metric_columns)
        .copy()
    )

    output_directory = os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        get_ranking_evaluation_result_directory(
            token_score_type
        ),
        model_name,
    )
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(
        output_directory,
        CLEAR_RESULT_FILENAMES[aggregation_method],
    )
    clear_metric_dataframe.to_csv(
        output_path,
        index=False,
        encoding='utf-8',
    )

    return output_path


def save_method_results(
        dataset_name: str,
        aggregation_method: str,
        ranking_strategy: RankingStrategy,
        release_pairs: Sequence[Tuple[str, str]],
        model_name: str,
        token_score_type: str,
) -> pd.DataFrame:

    method_output_directory = os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        get_ranking_evaluation_result_directory(token_score_type),
        model_name,
        aggregation_method,
    )
    os.makedirs(method_output_directory, exist_ok=True)

    evaluation_rows: List[dict] = []

    for train_release, test_release in release_pairs:
        test_dataframe = prepare_test_dataframe(
            dataset_name=dataset_name,
            train_release=train_release,
            test_release=test_release,
            aggregation_method=aggregation_method,
            model_name=model_name,
            token_score_type=token_score_type,
        )

        ranking_indices = apply_ranking_strategy(
            test_dataframe,
            ranking_strategy,
        )
        metrics = calculate_ranking_metrics(
            test_dataframe,
            ranking_indices,
        )

        evaluation_rows.append(
            build_evaluation_row(
                train_release=train_release,
                test_release=test_release,
                aggregation_method=aggregation_method,
                strategy_name=ranking_strategy.name,
                token_score_type=token_score_type,
                metrics=metrics,
            )
        )

        ranked_dataframe = (
            test_dataframe.iloc[ranking_indices]
            .copy()
            .reset_index(drop=True)
        )
        ranked_dataframe.insert(
            0,
            'rank',
            np.arange(1, len(ranked_dataframe) + 1),
        )
        ranked_dataframe['ranking_strategy'] = ranking_strategy.name

        ranking_columns = [
            'rank',
            'line_id',
            'filename',
            'line_number',
            'original_line_number',
            'code_line',
            'line-label',
            'suspicion_score',
            'line_counterfactual_token_count',
            'line_counterfactual_unique_token_count',
            'contains_line_counterfactual_token',
            'aggregation_method',
            'ranking_strategy',
            'token_score_type',
        ]
        available_columns = [
            column
            for column in ranking_columns
            if column in ranked_dataframe.columns
        ]

        ranking_output_path = os.path.join(
            method_output_directory,
            f'{test_release}_{aggregation_method}_best_ranking.csv',
        )
        ranked_dataframe[available_columns].to_csv(
            ranking_output_path,
            index=False,
            encoding='utf-8',
        )

    evaluation_columns = [
        'train_release',
        'test_release',
        'aggregation_method',
        'ranking_strategy',
        'token_score_type',
        'tp',
        'fp',
        'fn',
        'tn',
        'precision',
        'recall',
        'far',
        'd2h',
        'mcc',
        'f1',
        'auc',
        'ifa',
        *[
            f'recall_{percentage}'
            for percentage in RECALL_PERCENTAGES
        ],
        'effort@20%recall',
    ]

    evaluation_dataframe = pd.DataFrame(
        evaluation_rows
    ).reindex(columns=evaluation_columns)

    evaluation_output_path = os.path.join(
        method_output_directory,
        f'{aggregation_method}_best_strategy_evaluation.csv',
    )
    evaluation_dataframe.to_csv(
        evaluation_output_path,
        index=False,
        encoding='utf-8',
    )

    save_clear_method_metric_csv(
        dataset_name=dataset_name,
        aggregation_method=aggregation_method,
        evaluation_dataframe=evaluation_dataframe,
        model_name=model_name,
        token_score_type=token_score_type,
    )

    metadata_output_path = os.path.join(
        method_output_directory,
        f'{aggregation_method}_best_strategy.json',
    )
    with open(
            metadata_output_path,
            'w',
            encoding='utf-8',
    ) as metadata_file:
        json.dump(
            {
                'dataset_name': dataset_name,
                'aggregation_method': aggregation_method,
                'token_score_type': token_score_type,
                'ranking_strategy': ranking_strategy.name,
                'ablation_removed_components': [
                    'file_level_prediction',
                    'file_level_counterfactual_tokens',
                ],
            },
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    return evaluation_dataframe


def collect_release_pairs(
        projects: dict,
) -> List[Tuple[str, str]]:

    release_pairs: List[Tuple[str, str]] = []

    for project_name, releases in projects.items():
        for release_index in range(1, len(releases)):
            release_pairs.append((
                releases[release_index - 1],
                releases[release_index],
            ))

    return release_pairs


def main() -> None:

    datasets_projects = {
        # 'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }
    selected_model_name = DEFAULT_FILE_MODEL_NAME

    if len(RANKING_STRATEGIES) != 1:
        raise ValueError(
            'Exactly one ranking strategy must be configured.'
        )

    ranking_strategy = RANKING_STRATEGIES[0]

    for token_score_type in TOKEN_SCORE_TYPES_TO_EVALUATE:
        validate_token_score_type(token_score_type)

        for dataset_name, projects in datasets_projects.items():
            release_pairs = collect_release_pairs(projects)

            if not release_pairs:
                raise ValueError(
                    f'No adjacent release pairs are available for '
                    f'{dataset_name}.'
                )

            method_summary_rows = []

            for aggregation_method in CLEAR_LINE_SCORE_METHODS:
                evaluation_dataframe = save_method_results(
                    dataset_name=dataset_name,
                    aggregation_method=aggregation_method,
                    ranking_strategy=ranking_strategy,
                    release_pairs=release_pairs,
                    model_name=selected_model_name,
                    token_score_type=token_score_type,
                )

                method_summary_rows.append({
                    'aggregation_method': aggregation_method,
                    'token_score_type': token_score_type,
                    'ranking_strategy': ranking_strategy.name,
                    'mean_recall_20': float(
                        evaluation_dataframe['recall_20'].mean()
                    ),
                    'median_recall_20': float(
                        evaluation_dataframe['recall_20'].median()
                    ),
                    'mean_ifa': float(
                        evaluation_dataframe['ifa'].mean()
                    ),
                    'mean_effort@20%recall': float(
                        evaluation_dataframe[
                            'effort@20%recall'
                        ].mean()
                    ),
                })

            summary_output_directory = os.path.join(
                DATA_ROOT,
                dataset_name,
                'results',
                get_ranking_evaluation_result_directory(
                    token_score_type
                ),
                selected_model_name,
            )
            os.makedirs(summary_output_directory, exist_ok=True)

            strategy_metadata_path = os.path.join(
                summary_output_directory,
                'ranking_strategy.json',
            )
            with open(
                    strategy_metadata_path,
                    'w',
                    encoding='utf-8',
            ) as strategy_file:
                json.dump(
                    {
                        'dataset_name': dataset_name,
                        'token_score_type': token_score_type,
                        'ranking_strategy': ranking_strategy.name,
                        'sort_columns': list(
                            ranking_strategy.sort_columns
                        ),
                        'ablation_removed_components': [
                            'file_level_prediction',
                            'file_level_counterfactual_tokens',
                        ],
                    },
                    strategy_file,
                    indent=2,
                    ensure_ascii=False,
                )

            method_summary_path = os.path.join(
                summary_output_directory,
                'method_summary.csv',
            )
            pd.DataFrame(method_summary_rows).to_csv(
                method_summary_path,
                index=False,
                encoding='utf-8',
            )

            print(
                f'[{dataset_name}][{token_score_type}] '
                f'Saved results to {summary_output_directory}'
            )


if __name__ == '__main__':
    main()
