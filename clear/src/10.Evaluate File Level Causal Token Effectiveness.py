import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clear.src.my_utils.helper import linedp_projects

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.normpath(
    os.path.join(SCRIPT_DIRECTORY, '..', 'Data')
)

DATASET_NAME = 'linedp_dataset'
MODEL_NAME = 'logistic_regression'
CLEAR_AGGREGATION_METHODS = (
    'sum',
    'max',
    'top3avg',
    'top2avg',
)
AGGREGATION_METHOD = None
FILE_DECISION_THRESHOLD = 0.5
EFFORT_RATIO = 0.2

OUTPUT_DIRECTORY_NAME = 'file_level_causal_token_effectiveness'
OUTPUT_FILENAME = None

TOKEN_VECTORIZER = CountVectorizer(lowercase=False)
TOKENIZER = TOKEN_VECTORIZER.build_tokenizer()


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def load_json_dictionary(input_path: str) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'JSON file not found: {input_path}')

    with open(input_path, 'r', encoding='utf-8') as input_file:
        data = json.load(input_file)

    if not isinstance(data, dict):
        raise ValueError(f'Expected a JSON object in {input_path}.')

    return data


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


def get_file_prediction_path(
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


def get_output_directory(
        dataset_name: str,
        model_name: str,
) -> str:
    return os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        OUTPUT_DIRECTORY_NAME,
        model_name,
    )


def load_preprocessed_line_data(
        dataset_name: str,
        release: str,
) -> pd.DataFrame:
    input_path = get_preprocessed_line_path(dataset_name, release)
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'Preprocessed line data not found: {input_path}'
        )

    dataframe = pd.read_csv(input_path, encoding='utf-8')
    required_columns = {'filename', 'code_line', 'line-label'}
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

    invalid_labels = set(dataframe['line-label'].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            'Line-level labels must be binary values 0 or 1. '
            f'Invalid labels: {sorted(invalid_labels)}'
        )

    dataframe['_line_original_order'] = np.arange(
        len(dataframe),
        dtype=int,
    )
    return dataframe


def load_file_predictions(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'File prediction file not found: {input_path}'
        )

    dataframe = pd.read_csv(input_path, encoding='utf-8')
    required_columns = {'filename', 'predicted_bug_prob'}
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

    if dataframe['filename'].duplicated().any():
        duplicate_files = (
            dataframe.loc[
                dataframe['filename'].duplicated(),
                'filename',
            ]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            'Duplicate file prediction rows were found: '
            f'{duplicate_files[:10]}'
        )

    dataframe['predicted_file_bug'] = (
            dataframe['predicted_bug_prob']
            >= FILE_DECISION_THRESHOLD
    ).astype(int)

    dataframe = dataframe.rename(
        columns={'predicted_bug_prob': 'file_pred_prob'}
    )

    return dataframe[
        ['filename', 'file_pred_prob', 'predicted_file_bug']
    ]


def contains_any_file_counterfactual_token(
        code_lines: Iterable[str],
        file_counterfactual_tokens: set,
) -> int:
    for code_line in code_lines:
        tokens = TOKENIZER(str(code_line))
        if any(
                token in file_counterfactual_tokens
                for token in tokens
        ):
            return 1
    return 0


def build_file_level_dataframe(
        dataset_name: str,
        train_release: str,
        test_release: str,
        model_name: str,
) -> pd.DataFrame:
    line_dataframe = load_preprocessed_line_data(
        dataset_name,
        test_release,
    )

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
    file_counterfactual_tokens = set(
        file_counterfactual_frequencies
    )

    grouped_rows: List[dict] = []
    for filename, file_lines in line_dataframe.groupby(
            'filename',
            sort=False,
    ):
        grouped_rows.append({
            'filename': str(filename),
            'file_label': int(file_lines['line-label'].max()),
            'file_contains_file_counterfactual_token': (
                contains_any_file_counterfactual_token(
                    file_lines['code_line'],
                    file_counterfactual_tokens,
                )
            ),
            '_file_original_order': int(
                file_lines['_line_original_order'].min()
            ),
            'line_count': int(len(file_lines)),
            'defective_line_count': int(
                file_lines['line-label'].sum()
            ),
        })

    file_dataframe = pd.DataFrame(grouped_rows)
    prediction_dataframe = load_file_predictions(
        get_file_prediction_path(
            dataset_name,
            test_release,
            model_name,
        )
    )

    file_dataframe = file_dataframe.merge(
        prediction_dataframe,
        on='filename',
        how='left',
        validate='one_to_one',
    )

    missing_prediction_mask = (
            file_dataframe['file_pred_prob'].isna()
            | file_dataframe['predicted_file_bug'].isna()
    )
    if missing_prediction_mask.any():
        missing_files = file_dataframe.loc[
            missing_prediction_mask,
            'filename',
        ].tolist()
        raise ValueError(
            'Missing file-level predictions for target files: '
            f'{missing_files[:10]}'
        )

    file_dataframe['predicted_file_bug'] = (
        file_dataframe['predicted_file_bug'].astype(int)
    )
    file_dataframe[
        'file_contains_file_counterfactual_token'
    ] = file_dataframe[
        'file_contains_file_counterfactual_token'
    ].astype(int)

    return file_dataframe


def rank_files(
        file_dataframe: pd.DataFrame,
        use_file_causal_token: bool,
) -> pd.DataFrame:
    sort_columns = ['predicted_file_bug']
    if use_file_causal_token:
        sort_columns.append(
            'file_contains_file_counterfactual_token'
        )
    sort_columns.extend([
        'file_pred_prob',
        '_file_original_order',
    ])

    ascending = [False] * len(sort_columns)
    ascending[-1] = True

    ranked_dataframe = (
        file_dataframe.sort_values(
            by=sort_columns,
            ascending=ascending,
            kind='mergesort',
        )
        .reset_index(drop=True)
        .copy()
    )
    ranked_dataframe.insert(
        0,
        'file_rank',
        np.arange(1, len(ranked_dataframe) + 1, dtype=int),
    )
    return ranked_dataframe


def calculate_file_ranking_metrics(
        ranked_dataframe: pd.DataFrame,
) -> dict:
    labels = ranked_dataframe['file_label'].to_numpy(dtype=int)
    total_files = len(labels)
    total_defective_files = int(labels.sum())

    if total_files == 0:
        raise ValueError('Cannot evaluate an empty file ranking.')

    positive_positions = np.flatnonzero(labels == 1)
    file_ifa = (
        int(positive_positions[0])
        if len(positive_positions) > 0
        else total_files
    )

    cumulative_defective_files = np.cumsum(labels)
    inspected_count = max(1, int(total_files * EFFORT_RATIO))
    inspected_count = min(inspected_count, total_files)
    found_at_20 = int(
        cumulative_defective_files[inspected_count - 1]
    )
    file_recall_20 = safe_divide(
        found_at_20,
        total_defective_files,
    )

    if total_defective_files == 0:
        file_effort_at_20_recall = 0.0
    else:
        required_defective_files = max(
            1,
            int(math.ceil(total_defective_files * 0.2)),
        )
        target_positions = np.flatnonzero(
            cumulative_defective_files >= required_defective_files
        )
        file_effort_at_20_recall = (
            1.0
            if len(target_positions) == 0
            else float((target_positions[0] + 1) / total_files)
        )

    if len(np.unique(labels)) == 2:
        rank_scores = np.linspace(
            1.0,
            0.0,
            num=total_files,
            endpoint=False,
        )
        file_auc = float(roc_auc_score(labels, rank_scores))
    else:
        file_auc = float('nan')

    return {
        'file_ifa': file_ifa,
        'first_defective_file_rank': file_ifa + 1,
        'file_recall_20': file_recall_20,
        'file_effort@20%recall': file_effort_at_20_recall,
        'file_auc': file_auc,
        'total_files': total_files,
        'total_defective_files': total_defective_files,
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


def save_ranking(
        ranked_dataframe: pd.DataFrame,
        output_path: str,
        ranking_name: str,
) -> None:
    output_dataframe = ranked_dataframe.copy()
    output_dataframe['file_ranking_configuration'] = ranking_name
    output_columns = [
        'file_rank',
        'filename',
        'file_label',
        'predicted_file_bug',
        'file_pred_prob',
        'file_contains_file_counterfactual_token',
        'line_count',
        'defective_line_count',
        'file_ranking_configuration',
    ]
    output_dataframe[output_columns].to_csv(
        output_path,
        index=False,
        encoding='utf-8',
    )


def evaluate_release_pair(
        dataset_name: str,
        train_release: str,
        test_release: str,
        model_name: str,
        output_directory: str,
) -> dict:
    file_dataframe = build_file_level_dataframe(
        dataset_name=dataset_name,
        train_release=train_release,
        test_release=test_release,
        model_name=model_name,
    )

    full_ranking = rank_files(
        file_dataframe,
        use_file_causal_token=True,
    )
    without_file_token_ranking = rank_files(
        file_dataframe,
        use_file_causal_token=False,
    )

    full_metrics = calculate_file_ranking_metrics(full_ranking)
    without_metrics = calculate_file_ranking_metrics(
        without_file_token_ranking
    )

    ranking_output_directory = os.path.join(
        output_directory,
        AGGREGATION_METHOD,
    )
    os.makedirs(ranking_output_directory, exist_ok=True)

    save_ranking(
        full_ranking,
        os.path.join(
            ranking_output_directory,
            f'{test_release}_full_file_ranking.csv',
        ),
        'with_file_level_causal_token',
    )
    save_ranking(
        without_file_token_ranking,
        os.path.join(
            ranking_output_directory,
            f'{test_release}_without_file_causal_token_'
            'file_ranking.csv',
        ),
        'without_file_level_causal_token',
    )

    return {
        'train_release': train_release,
        'test_release': test_release,
        'release': test_release,
        'aggregation_method': AGGREGATION_METHOD,
        'full_file_ifa': full_metrics['file_ifa'],
        'without_file_causal_token_file_ifa': (
            without_metrics['file_ifa']
        ),
        'file_ifa_gain': (
                without_metrics['file_ifa']
                - full_metrics['file_ifa']
        ),
        'full_first_defective_file_rank': (
            full_metrics['first_defective_file_rank']
        ),
        'without_file_causal_token_first_defective_file_rank': (
            without_metrics['first_defective_file_rank']
        ),
        'full_file_recall_20': full_metrics['file_recall_20'],
        'without_file_causal_token_file_recall_20': (
            without_metrics['file_recall_20']
        ),
        'full_file_effort@20%recall': (
            full_metrics['file_effort@20%recall']
        ),
        'without_file_causal_token_file_effort@20%recall': (
            without_metrics['file_effort@20%recall']
        ),
        'full_file_auc': full_metrics['file_auc'],
        'without_file_causal_token_file_auc': (
            without_metrics['file_auc']
        ),
        'total_files': full_metrics['total_files'],
        'total_defective_files': (
            full_metrics['total_defective_files']
        ),
    }


def run_one_aggregation_method(
        aggregation_method: str,
) -> None:
    global AGGREGATION_METHOD
    global OUTPUT_FILENAME

    AGGREGATION_METHOD = aggregation_method
    OUTPUT_FILENAME = (
        f"CLEAR-{AGGREGATION_METHOD.upper()}_file_level_IFA.csv"
    )

    release_pairs = collect_release_pairs(linedp_projects)
    if not release_pairs:
        raise ValueError('No adjacent release pairs are available.')

    output_directory = get_output_directory(
        DATASET_NAME,
        MODEL_NAME,
    )
    os.makedirs(output_directory, exist_ok=True)

    result_rows: List[dict] = []
    for train_release, test_release in tqdm(
            release_pairs,
            desc=f'Evaluating file-level causal-token effect [{AGGREGATION_METHOD}]',
            unit='release',
            dynamic_ncols=True,
    ):
        result_rows.append(
            evaluate_release_pair(
                dataset_name=DATASET_NAME,
                train_release=train_release,
                test_release=test_release,
                model_name=MODEL_NAME,
                output_directory=output_directory,
            )
        )

    result_dataframe = pd.DataFrame(result_rows)
    output_path = os.path.join(
        output_directory,
        OUTPUT_FILENAME,
    )
    result_dataframe.to_csv(
        output_path,
        index=False,
        encoding='utf-8',
    )

    metadata_path = os.path.join(
        output_directory,
        (
            f'CLEAR-{AGGREGATION_METHOD.upper()}'
            '_file_level_IFA_metadata.json'
        ),
    )
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(
            {
                'dataset_name': DATASET_NAME,
                'model_name': MODEL_NAME,
                'aggregation_method': AGGREGATION_METHOD,
                'file_decision_threshold': FILE_DECISION_THRESHOLD,
                'full_file_ranking_keys': [
                    'predicted_file_bug',
                    'file_contains_file_counterfactual_token',
                    'file_pred_prob',
                    'stable_source_file_order',
                ],
                'ablated_file_ranking_keys': [
                    'predicted_file_bug',
                    'file_pred_prob',
                    'stable_source_file_order',
                ],
                'file_label_definition': (
                    'A file is defective when at least one line has '
                    'line-label equal to 1.'
                ),
                'file_ifa_definition': (
                    'The number of non-defective files before the '
                    'first defective file.'
                ),
                'line_level_causal_tokens_used': False,
                'line_level_suspiciousness_used': False,
                'changed_component': (
                    'file_level_causal_token_context'
                ),
                'metric_scope': 'file_level',
                'release_pairs_changed': False,
            },
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    print(f'Saved file-level causal-token results to {output_path}')


def main() -> None:
    for aggregation_method in CLEAR_AGGREGATION_METHODS:
        run_one_aggregation_method(
            aggregation_method
        )


if __name__ == '__main__':
    main()
