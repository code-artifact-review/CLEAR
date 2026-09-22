import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clear.src.my_utils.helper import linedp_projects

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(
    os.path.join(SCRIPT_DIRECTORY, '..', 'Data')
)

DEFAULT_FILE_MODEL_NAME = 'logistic_regression'
DEFAULT_FILE_DECISION_THRESHOLD = 0.5
DEFAULT_EFFORT_RATIO = 0.2

CLEAR_LINE_SCORE_METHODS = (
    'sum',
    'max',
    'top3avg',
    'top2avg',
)

RECALL_PERCENTAGES = tuple(range(0, 101, 10))

CLEAR_RESULT_FILENAMES = {
    'max': 'CLEAR-MAX.csv',
    'top3avg': 'CLEAR-TOP3AVG.csv',
    'sum': 'CLEAR-SUM.csv',
    'top2avg': 'CLEAR-TOP2AVG.csv',
}

TOKEN_SCORE_TYPES_TO_EVALUATE = (
    'confidence_enhanced',
)

LINE_COUNTERFACTUAL_RESULT_DIRECTORIES = {
    'confidence_enhanced': 'line_counterfactual_analysis',
    'suspiciousness_only': (
        'line_counterfactual_analysis_without_confidence'
    ),
}

TOKEN_VECTORIZER = CountVectorizer(lowercase=False)
TOKENIZER = TOKEN_VECTORIZER.build_tokenizer()


@dataclass(frozen=True)
class RankingStrategy:
    name: str
    sort_columns: Tuple[str, ...]


@dataclass(frozen=True)
class CausalTokenAblation:
    name: str
    output_directory: str
    removed_components: Tuple[str, ...]
    ranking_strategy: RankingStrategy


def build_causal_token_ablations(
) -> Tuple[CausalTokenAblation, ...]:

    return (
        CausalTokenAblation(
            name='without_line_counterfactual_token',
            output_directory=(
                'ranking_evaluation_'
                'without_line_counterfactual_token'
            ),
            removed_components=(
                'line_level_counterfactual_token',
            ),
            ranking_strategy=RankingStrategy(
                name=(
                    'global_file_context_then_'
                    'suspiciousness'
                ),
                sort_columns=(
                    'file_contains_file_counterfactual_token',
                    'suspicion_score',
                    'file_pred_prob',
                ),
            ),
        ),
        CausalTokenAblation(
            name='without_file_counterfactual_token',
            output_directory=(
                'ranking_evaluation_'
                'without_file_counterfactual_token'
            ),
            removed_components=(
                'file_level_counterfactual_token_context',
            ),
            ranking_strategy=RankingStrategy(
                name=(
                    'global_line_causal_then_'
                    'suspiciousness'
                ),
                sort_columns=(
                    'contains_line_counterfactual_token',
                    'suspicion_score',
                    'file_pred_prob',
                ),
            ),
        ),
        CausalTokenAblation(
            name=(
                'without_line_and_file_'
                'counterfactual_tokens'
            ),
            output_directory=(
                'ranking_evaluation_without_line_and_'
                'file_counterfactual_tokens'
            ),
            removed_components=(
                'line_level_counterfactual_token',
                'file_level_counterfactual_token_context',
            ),
            ranking_strategy=RankingStrategy(
                name=(
                    'global_suspiciousness_only_'
                    'after_file_prediction'
                ),
                sort_columns=(
                    'suspicion_score',
                    'file_pred_prob',
                ),
            ),
        ),
    )


CAUSAL_TOKEN_ABLATIONS = build_causal_token_ablations()


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def load_json_dictionary(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'JSON file not found: {path}')

    with open(path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(
            f'Expected a JSON object in {path}.'
        )

    return data


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


def get_line_counterfactual_result_directory(
        token_score_type: str,
) -> str:
    if token_score_type not in LINE_COUNTERFACTUAL_RESULT_DIRECTORIES:
        raise ValueError(
            f'Unsupported token score type: {token_score_type}'
        )

    return LINE_COUNTERFACTUAL_RESULT_DIRECTORIES[
        token_score_type
    ]


def get_test_file_prediction_path(
        dataset_name: str,
        test_release: str,
        model_name: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        'file_prediction_results',
        model_name,
        f'{test_release}_file_predictions.csv',
    )


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


def get_file_counterfactual_frequency_path(
        dataset_name: str,
        train_release: str,
        model_name: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        'file_counterfactual_analysis',
        model_name,
        f'{train_release}_validation_'
        'file_counterfactual_token_frequency.json',
    )


def load_file_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'File prediction file not found: {path}'
        )

    dataframe = pd.read_csv(path, encoding='utf-8')

    required_columns = {
        'filename',
        'predicted_bug_prob',
    }
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required file prediction columns: '
            f'{sorted(missing_columns)}'
        )

    dataframe = dataframe.copy()
    dataframe['filename'] = dataframe['filename'].astype(str)
    dataframe['predicted_bug_prob'] = pd.to_numeric(
        dataframe['predicted_bug_prob'],
        errors='raise',
    )
    dataframe['predicted_bug'] = (
        dataframe['predicted_bug_prob']
        >= DEFAULT_FILE_DECISION_THRESHOLD
    ).astype(int)

    return dataframe


def add_file_prediction_columns(
        line_dataframe: pd.DataFrame,
        file_prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:

    probability_map = dict(zip(
        file_prediction_dataframe['filename'],
        file_prediction_dataframe['predicted_bug_prob'],
    ))
    prediction_map = dict(zip(
        file_prediction_dataframe['filename'],
        file_prediction_dataframe['predicted_bug'],
    ))

    output_dataframe = line_dataframe.copy()
    output_dataframe['file_pred_prob'] = (
        output_dataframe['filename'].map(probability_map)
    )
    output_dataframe['predicted_file_bug'] = (
        output_dataframe['filename'].map(prediction_map)
    )

    missing_mask = (
        output_dataframe['file_pred_prob'].isna()
        | output_dataframe['predicted_file_bug'].isna()
    )
    if missing_mask.any():
        missing_files = sorted(
            output_dataframe.loc[
                missing_mask,
                'filename',
            ].unique()
        )
        raise ValueError(
            'Missing file-level predictions for source files: '
            f'{missing_files[:10]}'
        )

    output_dataframe['predicted_file_bug'] = (
        output_dataframe['predicted_file_bug'].astype(int)
    )

    return output_dataframe


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


def add_counterfactual_features(
        dataframe: pd.DataFrame,
        line_counterfactual_tokens: set,
        file_counterfactual_frequencies: Dict[str, int],
) -> pd.DataFrame:

    output_dataframe = dataframe.copy()

    line_occurrence_counts: List[int] = []
    line_unique_counts: List[int] = []
    file_occurrence_counts: List[int] = []
    file_unique_counts: List[int] = []
    file_frequency_weights: List[float] = []

    file_counterfactual_tokens = set(
        file_counterfactual_frequencies
    )

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
        file_occurrence_counts.append(
            sum(
                1
                for token in tokens
                if token in file_counterfactual_tokens
            )
        )
        file_unique_counts.append(
            len(unique_tokens & file_counterfactual_tokens)
        )
        file_frequency_weights.append(
            float(sum(
                file_counterfactual_frequencies.get(token, 0)
                for token in tokens
            ))
        )

    output_dataframe[
        'line_counterfactual_token_count'
    ] = line_occurrence_counts
    output_dataframe[
        'line_counterfactual_unique_token_count'
    ] = line_unique_counts
    output_dataframe[
        'file_counterfactual_token_count'
    ] = file_occurrence_counts
    output_dataframe[
        'file_counterfactual_unique_token_count'
    ] = file_unique_counts
    output_dataframe[
        'file_counterfactual_frequency_weight'
    ] = file_frequency_weights

    output_dataframe[
        'contains_line_counterfactual_token'
    ] = (
        output_dataframe[
            'line_counterfactual_token_count'
        ] > 0
    ).astype(int)

    output_dataframe[
        'contains_file_counterfactual_token'
    ] = (
        output_dataframe[
            'file_counterfactual_token_count'
        ] > 0
    ).astype(int)

    output_dataframe[
        'contains_both_counterfactual_token_types'
    ] = (
        (
            output_dataframe[
                'contains_line_counterfactual_token'
            ] == 1
        )
        & (
            output_dataframe[
                'contains_file_counterfactual_token'
            ] == 1
        )
    ).astype(int)

    output_dataframe[
        'total_counterfactual_token_count'
    ] = (
        output_dataframe[
            'line_counterfactual_token_count'
        ]
        + output_dataframe[
            'file_counterfactual_token_count'
        ]
    )

    output_dataframe[
        'file_contains_file_counterfactual_token'
    ] = (
        output_dataframe.groupby('filename')[
            'contains_file_counterfactual_token'
        ].transform('max').astype(int)
    )

    return output_dataframe


def rank_indices_by_columns(
        dataframe: pd.DataFrame,
        sort_columns: Sequence[str],
) -> np.ndarray:

    effective_columns = [
        'predicted_file_bug',
        *sort_columns,
    ]

    missing_columns = (
        set(effective_columns) - set(dataframe.columns)
    )
    if missing_columns:
        raise ValueError(
            'Missing ranking columns: '
            f'{sorted(missing_columns)}'
        )

    stable_order = dataframe[
        '_original_order'
    ].to_numpy(dtype=int)

    lexicographic_keys: List[np.ndarray] = [
        stable_order
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

    return {
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
    }


def prepare_test_dataframe(
        dataset_name: str,
        train_release: str,
        test_release: str,
        aggregation_method: str,
        model_name: str,
        token_score_type: str,
) -> pd.DataFrame:

    file_counterfactual_frequencies = {
        str(token): int(frequency)
        for token, frequency in load_json_dictionary(
            get_file_counterfactual_frequency_path(
                dataset_name,
                train_release,
                model_name,
            )
        ).items()
    }

    line_counterfactual_tokens = load_json_token_set(
        get_line_counterfactual_token_path(
            dataset_name=dataset_name,
            train_release=train_release,
            aggregation_method=aggregation_method,
            model_name=model_name,
            token_score_type=token_score_type,
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

    if 'file_pred_prob' in dataframe.columns:
        dataframe['file_pred_prob'] = pd.to_numeric(
            dataframe['file_pred_prob'],
            errors='raise',
        )
        dataframe['predicted_file_bug'] = (
            dataframe['file_pred_prob']
            >= DEFAULT_FILE_DECISION_THRESHOLD
        ).astype(int)
    else:
        test_file_predictions = load_file_predictions(
            get_test_file_prediction_path(
                dataset_name,
                test_release,
                model_name,
            )
        )
        dataframe = add_file_prediction_columns(
            dataframe,
            test_file_predictions,
        )

    dataframe = add_counterfactual_features(
        dataframe,
        line_counterfactual_tokens,
        file_counterfactual_frequencies,
    )
    dataframe = add_line_identifiers(dataframe)
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


def collect_release_pairs(
        projects: dict,
) -> List[Tuple[str, str]]:

    release_pairs: List[Tuple[str, str]] = []

    for releases in projects.values():
        for release_index in range(1, len(releases)):
            release_pairs.append((
                releases[release_index - 1],
                releases[release_index],
            ))

    return release_pairs


def get_ablation_output_directory(
        dataset_name: str,
        ablation: CausalTokenAblation,
        model_name: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        ablation.output_directory,
        model_name,
    )


def save_ablation_metric_csv(
        dataset_name: str,
        aggregation_method: str,
        evaluation_dataframe: pd.DataFrame,
        model_name: str,
        ablation: CausalTokenAblation,
) -> str:

    if aggregation_method not in CLEAR_RESULT_FILENAMES:
        raise ValueError(
            'Unsupported CLEAR aggregation method: '
            f'{aggregation_method}'
        )

    output_columns = [
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

    output_dataframe = (
        evaluation_dataframe.rename(
            columns={'test_release': 'release'}
        )
        .reindex(columns=output_columns)
        .copy()
    )

    output_directory = get_ablation_output_directory(
        dataset_name,
        ablation,
        model_name,
    )
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(
        output_directory,
        CLEAR_RESULT_FILENAMES[aggregation_method],
    )

    output_dataframe.to_csv(
        output_path,
        index=False,
        encoding='utf-8',
    )

    return output_path


def evaluate_and_save_ablation(
        dataset_name: str,
        aggregation_method: str,
        release_pairs: Sequence[Tuple[str, str]],
        model_name: str,
        token_score_type: str,
        ablation: CausalTokenAblation,
) -> pd.DataFrame:

    method_output_directory = os.path.join(
        get_ablation_output_directory(
            dataset_name,
            ablation,
            model_name,
        ),
        aggregation_method,
    )
    os.makedirs(method_output_directory, exist_ok=True)

    evaluation_rows: List[dict] = []
    strategy = ablation.ranking_strategy

    for train_release, test_release in release_pairs:
        test_dataframe = prepare_test_dataframe(
            dataset_name=dataset_name,
            train_release=train_release,
            test_release=test_release,
            aggregation_method=aggregation_method,
            model_name=model_name,
            token_score_type=token_score_type,
        )

        ranking_indices = rank_indices_by_columns(
            test_dataframe,
            strategy.sort_columns,
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
                strategy_name=strategy.name,
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
        ranked_dataframe[
            'ranking_strategy'
        ] = strategy.name
        ranked_dataframe[
            'causal_token_ablation'
        ] = ablation.name

        ranking_columns = [
            'rank',
            'line_id',
            'filename',
            'line_number',
            'original_line_number',
            'code_line',
            'line-label',
            'predicted_file_bug',
            'file_pred_prob',
            'suspicion_score',
            'line_counterfactual_token_count',
            'file_counterfactual_token_count',
            'total_counterfactual_token_count',
            'contains_line_counterfactual_token',
            'contains_file_counterfactual_token',
            'file_contains_file_counterfactual_token',
            'contains_both_counterfactual_token_types',
            'aggregation_method',
            'ranking_strategy',
            'causal_token_ablation',
            'token_score_type',
        ]

        available_columns = [
            column
            for column in ranking_columns
            if column in ranked_dataframe.columns
        ]

        ranking_output_path = os.path.join(
            method_output_directory,
            f'{test_release}_{aggregation_method}_'
            'best_ranking.csv',
        )

        ranked_dataframe[
            available_columns
        ].to_csv(
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
    ).reindex(
        columns=evaluation_columns
    )

    evaluation_output_path = os.path.join(
        method_output_directory,
        f'{aggregation_method}_ablation_evaluation.csv',
    )

    evaluation_dataframe.to_csv(
        evaluation_output_path,
        index=False,
        encoding='utf-8',
    )

    save_ablation_metric_csv(
        dataset_name=dataset_name,
        aggregation_method=aggregation_method,
        evaluation_dataframe=evaluation_dataframe,
        model_name=model_name,
        ablation=ablation,
    )

    metadata_output_path = os.path.join(
        method_output_directory,
        f'{aggregation_method}_ablation_metadata.json',
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
                'ablation_name': ablation.name,
                'removed_components': list(
                    ablation.removed_components
                ),
                'ranking_strategy': strategy.name,
                'ranking_sort_columns': list(
                    strategy.sort_columns
                ),
            },
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    return evaluation_dataframe


def main() -> None:
    datasets_projects = {
        'linedp_dataset': linedp_projects,
    }

    selected_model_name = DEFAULT_FILE_MODEL_NAME

    for token_score_type in TOKEN_SCORE_TYPES_TO_EVALUATE:
        for dataset_name, projects in datasets_projects.items():
            release_pairs = collect_release_pairs(
                projects
            )

            if not release_pairs:
                raise ValueError(
                    'No adjacent release pairs are available for '
                    f'{dataset_name}.'
                )

            summary_rows: List[dict] = []

            for ablation in CAUSAL_TOKEN_ABLATIONS:
                for aggregation_method in CLEAR_LINE_SCORE_METHODS:
                    evaluation_dataframe = (
                        evaluate_and_save_ablation(
                            dataset_name=dataset_name,
                            aggregation_method=(
                                aggregation_method
                            ),
                            release_pairs=release_pairs,
                            model_name=selected_model_name,
                            token_score_type=(
                                token_score_type
                            ),
                            ablation=ablation,
                        )
                    )

                    summary_rows.append({
                        'dataset_name': dataset_name,
                        'aggregation_method': (
                            aggregation_method
                        ),
                        'token_score_type': token_score_type,
                        'ablation_name': ablation.name,
                        'ranking_strategy': (
                            ablation.ranking_strategy.name
                        ),
                        'removed_components': '|'.join(
                            ablation.removed_components
                        ),
                        'evaluated_releases': int(
                            evaluation_dataframe[
                                'test_release'
                            ].nunique()
                        ),
                        'mean_ifa': float(
                            evaluation_dataframe[
                                'ifa'
                            ].mean()
                        ),
                        'median_ifa': float(
                            evaluation_dataframe[
                                'ifa'
                            ].median()
                        ),
                        'mean_recall_20': float(
                            evaluation_dataframe[
                                'recall_20'
                            ].mean()
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
                'causal_token_ablation_summary',
                selected_model_name,
            )
            os.makedirs(
                summary_output_directory,
                exist_ok=True,
            )

            summary_output_path = os.path.join(
                summary_output_directory,
                'CLEAR-ALL-AGGREGATIONS_causal_token_'
                'ablation_summary.csv',
            )

            pd.DataFrame(
                summary_rows
            ).to_csv(
                summary_output_path,
                index=False,
                encoding='utf-8',
            )

            print(
                f'[{dataset_name}] Saved results to '
                f'{summary_output_path}'
            )


if __name__ == '__main__':
    main()
