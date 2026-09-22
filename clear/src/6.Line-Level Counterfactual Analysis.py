import json
import os
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve
from tqdm import tqdm

from clear.src.my_utils.helper import (
    glance_projects,
    linedp_projects,
)
from file_level_prediction import (
    DEFAULT_FILE_MODEL_NAME,
    get_file_prediction_result_path,
)

DATA_ROOT = '../Data'

CLEAR_LINE_SCORE_METHODS = (
    'sum',
    'max',
    'top2avg',
    'top3avg',
)

TOKEN_VECTORIZER = CountVectorizer(lowercase=False)
TOKENIZER = TOKEN_VECTORIZER.build_tokenizer()

EXCLUDED_LINE_COUNTERFACTUAL_TOKENS = {
    '<str>',
    'str',
}


def load_preprocessed_line_data(
        dataset_name: str,
        release: str,
) -> pd.DataFrame:

    input_path = os.path.join(
        DATA_ROOT,
        dataset_name,
        'preprocessed_data',
        f'{release}.csv',
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'Preprocessed line data not found: {input_path}'
        )

    line_dataframe = pd.read_csv(input_path, encoding='utf-8')

    required_columns = {'filename', 'code_line', 'line-label'}
    missing_columns = required_columns - set(line_dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required columns in preprocessed line data: '
            f'{sorted(missing_columns)}'
        )

    if line_dataframe.empty:
        raise ValueError(
            f'Preprocessed line data is empty for release {release}.'
        )

    line_dataframe = line_dataframe.copy()
    line_dataframe['filename'] = line_dataframe['filename'].astype(str)
    line_dataframe['code_line'] = (
        line_dataframe['code_line'].fillna('').astype(str)
    )
    line_dataframe['line-label'] = pd.to_numeric(
        line_dataframe['line-label'],
        errors='raise',
    ).astype(int)

    invalid_labels = set(line_dataframe['line-label'].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            'Line-level labels must be binary values 0 or 1. '
            f'Invalid labels: {sorted(invalid_labels)}'
        )

    return line_dataframe


def load_confidence_enhanced_token_scores(
        dataset_name: str,
        train_release: str,
) -> Dict[str, float]:
    """Load CLEAR confidence-enhanced token risk scores."""

    score_path = os.path.join(
        DATA_ROOT,
        dataset_name,
        'token_scores',
        f'{train_release}_confidence_enhanced_token_scores.json',
    )

    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f'CLEAR token score file not found: {score_path}'
        )

    with open(score_path, 'r', encoding='utf-8') as score_file:
        token_scores = json.load(score_file)

    if not isinstance(token_scores, dict):
        raise ValueError(f'Invalid token score file: {score_path}')

    return {
        str(token): float(score)
        for token, score in token_scores.items()
    }


def load_suspiciousness_token_scores(
        dataset_name: str,
        train_release: str,
) -> Dict[str, float]:

    score_path = os.path.join(
        DATA_ROOT,
        dataset_name,
        'token_scores',
        f'{train_release}_suspiciousness_token_scores.json',
    )

    if not os.path.exists(score_path):
        raise FileNotFoundError(
            f'CLEAR suspiciousness token score file not found: {score_path}'
        )

    with open(score_path, 'r', encoding='utf-8') as score_file:
        token_scores = json.load(score_file)

    if not isinstance(token_scores, dict):
        raise ValueError(f'Invalid token score file: {score_path}')

    return {
        str(token): float(score)
        for token, score in token_scores.items()
    }


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
        f'Unknown CLEAR line score method: {aggregation_method}'
    )


def select_youden_threshold(
        line_labels: Iterable[int],
        line_scores: Iterable[float],
) -> float:
    """Select the finite line-risk threshold maximizing Youden's J."""

    labels = np.asarray(list(line_labels), dtype=int)
    scores = np.asarray(list(line_scores), dtype=float)

    if len(labels) != len(scores):
        raise ValueError(
            'line_labels and line_scores must have the same length.'
        )

    if len(labels) == 0:
        raise ValueError('Cannot select a threshold from empty data.')

    if not np.isfinite(scores).all():
        raise ValueError('Line scores contain NaN or infinite values.')

    if len(np.unique(labels)) < 2:
        raise ValueError(
            'Youden threshold selection requires both defective '
            'and clean code lines.'
        )

    false_positive_rates, true_positive_rates, thresholds = roc_curve(
        labels,
        scores,
    )
    youden_statistics = true_positive_rates - false_positive_rates

    finite_mask = np.isfinite(thresholds)
    finite_thresholds = thresholds[finite_mask]
    finite_youden_statistics = youden_statistics[finite_mask]

    if len(finite_thresholds) == 0:
        raise ValueError('No finite ROC threshold is available.')

    best_index = int(np.argmax(finite_youden_statistics))
    return float(finite_thresholds[best_index])


def identify_line_level_counterfactual_tokens(
        historical_dataframe: pd.DataFrame,
        token_risk_scores: Dict[str, float],
        aggregation_method: str,
        line_risk_threshold: float,
) -> Set[str]:

    counterfactual_token_set: Set[str] = set()
    score_column = f'clear_{aggregation_method}'

    required_columns = {
        'code_line',
        'line-label',
        score_column,
    }
    missing_columns = required_columns - set(historical_dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required columns for counterfactual analysis: '
            f'{sorted(missing_columns)}'
        )

    analysis_rows = historical_dataframe[
        [
            'code_line',
            'line-label',
            score_column,
        ]
    ].itertuples(index=False, name=None)

    for (
            code_line,
            line_label,
            original_line_score,
    ) in tqdm(
        analysis_rows,
        total=len(historical_dataframe),
        desc=(
                'Identifying line-level counterfactual tokens '
                f'({aggregation_method})'
        ),
        unit='line',
        leave=False,
        dynamic_ncols=True,
    ):
        original_line_score = float(original_line_score)
        line_label = int(line_label)

        if (
                line_label != 1
                or original_line_score < line_risk_threshold
        ):
            continue

        tokens = TOKENIZER(str(code_line))

        for token_index, token in enumerate(tokens):
            if token in EXCLUDED_LINE_COUNTERFACTUAL_TOKENS:
                continue

            counterfactual_tokens = (
                    tokens[:token_index]
                    + tokens[token_index + 1:]
            )
            counterfactual_line_score = calculate_line_risk_score(
                counterfactual_tokens,
                token_risk_scores,
                aggregation_method,
            )

            if counterfactual_line_score < line_risk_threshold:
                counterfactual_token_set.add(token)

    return counterfactual_token_set


def validate_file_prediction_mapping(
        line_dataframe: pd.DataFrame,
) -> None:

    missing_prediction_mask = (
            line_dataframe['file_pred_prob'].isna()
            | line_dataframe['predicted_file_bug'].isna()
    )

    if not missing_prediction_mask.any():
        return

    missing_files = sorted(
        line_dataframe.loc[
            missing_prediction_mask,
            'filename',
        ].astype(str).unique()
    )

    preview = missing_files[:10]
    suffix = (
        ''
        if len(missing_files) <= 10
        else f' ... and {len(missing_files) - 10} more'
    )

    raise ValueError(
        'Missing file-level predictions for files: '
        f'{preview}{suffix}'
    )


def add_file_prediction_columns(
        line_dataframe: pd.DataFrame,
        file_prediction_dataframe: pd.DataFrame,
        include_true_file_label: bool = False,
) -> pd.DataFrame:
    """Map file-level prediction results to all code lines."""

    required_file_columns = {
        'filename',
        'predicted_bug_prob',
        'predicted_bug',
    }
    if include_true_file_label:
        required_file_columns.add('true_bug')

    missing_file_columns = (
            required_file_columns - set(file_prediction_dataframe.columns)
    )
    if missing_file_columns:
        raise ValueError(
            'Missing required file prediction columns: '
            f'{sorted(missing_file_columns)}'
        )

    file_prediction_dataframe = file_prediction_dataframe.copy()
    file_prediction_dataframe['filename'] = (
        file_prediction_dataframe['filename'].astype(str)
    )
    file_prediction_dataframe['predicted_bug_prob'] = pd.to_numeric(
        file_prediction_dataframe['predicted_bug_prob'],
        errors='raise',
    )
    file_prediction_dataframe['predicted_bug'] = pd.to_numeric(
        file_prediction_dataframe['predicted_bug'],
        errors='raise',
    ).astype(int)

    invalid_file_predictions = (
            set(file_prediction_dataframe['predicted_bug'].unique())
            - {0, 1}
    )
    if invalid_file_predictions:
        raise ValueError(
            'File-level predictions must be binary values 0 or 1. '
            f'Invalid predictions: {sorted(invalid_file_predictions)}'
        )

    file_probability_map = dict(zip(
        file_prediction_dataframe['filename'],
        file_prediction_dataframe['predicted_bug_prob'],
    ))
    file_prediction_map = dict(zip(
        file_prediction_dataframe['filename'],
        file_prediction_dataframe['predicted_bug'],
    ))

    output_dataframe = line_dataframe.copy()
    output_dataframe['file_pred_prob'] = (
        output_dataframe['filename'].map(file_probability_map)
    )
    output_dataframe['predicted_file_bug'] = (
        output_dataframe['filename'].map(file_prediction_map)
    )

    if include_true_file_label:
        file_prediction_dataframe['true_bug'] = pd.to_numeric(
            file_prediction_dataframe['true_bug'],
            errors='raise',
        ).astype(int)
        true_file_label_map = dict(zip(
            file_prediction_dataframe['filename'],
            file_prediction_dataframe['true_bug'],
        ))
        output_dataframe['true_file_bug'] = (
            output_dataframe['filename'].map(true_file_label_map)
        )

    validate_file_prediction_mapping(output_dataframe)
    output_dataframe['predicted_file_bug'] = (
        output_dataframe['predicted_file_bug'].astype(int)
    )
    if include_true_file_label:
        if output_dataframe['true_file_bug'].isna().any():
            raise ValueError(
                'Missing true file-level labels for validation lines.'
            )
        output_dataframe['true_file_bug'] = (
            output_dataframe['true_file_bug'].astype(int)
        )

    return output_dataframe


def run_line_level_counterfactual_analysis(
        dataset_name: str,
        train_release: str,
        test_release: str,
        file_model_name: str = DEFAULT_FILE_MODEL_NAME,
        aggregation_methods: Tuple[str, ...] = CLEAR_LINE_SCORE_METHODS,
) -> None:
    """Run CLEAR line scoring and line-level counterfactual analysis."""

    token_risk_scores = load_confidence_enhanced_token_scores(
        dataset_name,
        train_release,
    )
    historical_dataframe = load_preprocessed_line_data(
        dataset_name,
        train_release,
    )
    test_dataframe = load_preprocessed_line_data(
        dataset_name,
        test_release,
    )

    if historical_dataframe.empty:
        raise ValueError(
            f'No historical code lines remain for release {train_release}.'
        )

    file_prediction_path = get_file_prediction_result_path(
        dataset_name=dataset_name,
        test_release=test_release,
        model_name=file_model_name,
    )
    if not os.path.exists(file_prediction_path):
        raise FileNotFoundError(
            'File-level prediction results not found: '
            f'{file_prediction_path}'
        )
    file_prediction_dataframe = pd.read_csv(
        file_prediction_path,
        encoding='utf-8',
    )
    test_dataframe = add_file_prediction_columns(
        test_dataframe,
        file_prediction_dataframe,
    )

    defective_line_count = int(
        (historical_dataframe['line-label'] == 1).sum()
    )
    total_line_count = len(historical_dataframe)
    line_defect_rate = (
        defective_line_count / total_line_count
        if total_line_count > 0
        else 0.0
    )

    prediction_directory = os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        'line_counterfactual_analysis',
        file_model_name,
        'predictions',
    )
    token_directory = os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        'line_counterfactual_analysis',
        file_model_name,
        'tokens',
    )
    os.makedirs(prediction_directory, exist_ok=True)
    os.makedirs(token_directory, exist_ok=True)

    for aggregation_method in aggregation_methods:
        if aggregation_method not in CLEAR_LINE_SCORE_METHODS:
            raise ValueError(
                f'Unsupported aggregation method: {aggregation_method}'
            )

        score_column = f'clear_{aggregation_method}'

        historical_dataframe_for_method = historical_dataframe.copy()
        historical_dataframe_for_method[score_column] = [
            calculate_line_risk_score(
                TOKENIZER(code_line),
                token_risk_scores,
                aggregation_method,
            )
            for code_line in historical_dataframe_for_method['code_line']
        ]

        line_risk_threshold = select_youden_threshold(
            historical_dataframe_for_method['line-label'],
            historical_dataframe_for_method[score_column],
        )

        counterfactual_token_set = (
            identify_line_level_counterfactual_tokens(
                historical_dataframe=historical_dataframe_for_method,
                token_risk_scores=token_risk_scores,
                aggregation_method=aggregation_method,
                line_risk_threshold=line_risk_threshold,
            )
        )

        test_output_dataframe = test_dataframe.copy()
        test_output_dataframe[score_column] = [
            calculate_line_risk_score(
                TOKENIZER(code_line),
                token_risk_scores,
                aggregation_method,
            )
            for code_line in test_output_dataframe['code_line']
        ]

        counterfactual_tokens_found: List[List[str]] = []
        counterfactual_token_counts: List[int] = []

        for code_line in test_output_dataframe['code_line']:
            tokens = TOKENIZER(str(code_line))

            found_tokens = [
                token
                for token in tokens
                if token in counterfactual_token_set
            ]

            counterfactual_tokens_found.append(found_tokens)
            counterfactual_token_counts.append(len(found_tokens))

        test_output_dataframe[
            'line_counterfactual_token_count'
        ] = counterfactual_token_counts
        test_output_dataframe[
            'line_counterfactual_tokens'
        ] = [
            json.dumps(tokens, ensure_ascii=False)
            for tokens in counterfactual_tokens_found
        ]
        test_output_dataframe['line_risk_threshold'] = (
            line_risk_threshold
        )
        test_output_dataframe['line_defect_rate'] = line_defect_rate
        test_output_dataframe['line_score_method'] = aggregation_method
        test_output_dataframe['file_model_name'] = file_model_name

        prediction_path = os.path.join(
            prediction_directory,
            f'{test_release}_clear_{aggregation_method}.csv',
        )
        test_output_dataframe.to_csv(
            prediction_path,
            index=False,
            encoding='utf-8',
        )

        token_path = os.path.join(
            token_directory,
            f'{train_release}_clear_{aggregation_method}_'
            'counterfactual_tokens.json',
        )
        with open(token_path, 'w', encoding='utf-8') as token_file:
            json.dump(
                sorted(counterfactual_token_set),
                token_file,
                indent=2,
                ensure_ascii=False,
            )

        metadata_path = os.path.join(
            token_directory,
            f'{train_release}_clear_{aggregation_method}_metadata.json',
        )
        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(
                {
                    'dataset_name': dataset_name,
                    'train_release': train_release,
                    'test_release': test_release,
                    'data_role_for_threshold_and_tokens': 'complete_previous_release',
                    'file_model_name': file_model_name,
                    'aggregation_method': aggregation_method,
                    'line_risk_threshold': line_risk_threshold,
                    'line_defect_rate': line_defect_rate,
                    'counterfactual_token_count': len(
                        counterfactual_token_set
                    ),
                    'token_intervention': (
                        'Remove one token occurrence from a code line.'
                    ),
                    'test_token_counting': (
                        'Count all matching token occurrences in each '
                        'code line.'
                    ),
                },
                metadata_file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f'[{train_release}] Selected {aggregation_method} '
            f'line threshold {line_risk_threshold:.6f} on the '
            'complete previous release.'
        )
        print(
            f'[{test_release}] Saved CLEAR {aggregation_method} '
            f'line results to {prediction_path}'
        )
        print(
            f'[{train_release}] Saved '
            f'{len(counterfactual_token_set)} line-level '
            f'counterfactual tokens to {token_path}'
        )


def run_line_level_counterfactual_analysis_without_confidence(
        dataset_name: str,
        train_release: str,
        test_release: str,
        file_model_name: str = DEFAULT_FILE_MODEL_NAME,
        aggregation_methods: Tuple[str, ...] = CLEAR_LINE_SCORE_METHODS,
) -> None:

    token_risk_scores = load_suspiciousness_token_scores(
        dataset_name,
        train_release,
    )
    historical_dataframe = load_preprocessed_line_data(
        dataset_name,
        train_release,
    )
    test_dataframe = load_preprocessed_line_data(
        dataset_name,
        test_release,
    )

    if historical_dataframe.empty:
        raise ValueError(
            f'No historical code lines remain for release {train_release}.'
        )

    file_prediction_path = get_file_prediction_result_path(
        dataset_name=dataset_name,
        test_release=test_release,
        model_name=file_model_name,
    )
    if not os.path.exists(file_prediction_path):
        raise FileNotFoundError(
            'File-level prediction results not found: '
            f'{file_prediction_path}'
        )
    file_prediction_dataframe = pd.read_csv(
        file_prediction_path,
        encoding='utf-8',
    )
    test_dataframe = add_file_prediction_columns(
        test_dataframe,
        file_prediction_dataframe,
    )

    defective_line_count = int(
        (historical_dataframe['line-label'] == 1).sum()
    )
    total_line_count = len(historical_dataframe)
    line_defect_rate = (
        defective_line_count / total_line_count
        if total_line_count > 0
        else 0.0
    )

    prediction_directory = os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        'line_counterfactual_analysis_without_confidence',
        file_model_name,
        'predictions',
    )
    token_directory = os.path.join(
        DATA_ROOT,
        dataset_name,
        'results',
        'line_counterfactual_analysis_without_confidence',
        file_model_name,
        'tokens',
    )
    os.makedirs(prediction_directory, exist_ok=True)
    os.makedirs(token_directory, exist_ok=True)

    for aggregation_method in aggregation_methods:
        if aggregation_method not in CLEAR_LINE_SCORE_METHODS:
            raise ValueError(
                f'Unsupported aggregation method: {aggregation_method}'
            )

        score_column = f'clear_{aggregation_method}'

        historical_dataframe_for_method = historical_dataframe.copy()
        historical_dataframe_for_method[score_column] = [
            calculate_line_risk_score(
                TOKENIZER(code_line),
                token_risk_scores,
                aggregation_method,
            )
            for code_line in historical_dataframe_for_method['code_line']
        ]

        line_risk_threshold = select_youden_threshold(
            historical_dataframe_for_method['line-label'],
            historical_dataframe_for_method[score_column],
        )

        counterfactual_token_set = (
            identify_line_level_counterfactual_tokens(
                historical_dataframe=historical_dataframe_for_method,
                token_risk_scores=token_risk_scores,
                aggregation_method=aggregation_method,
                line_risk_threshold=line_risk_threshold,
            )
        )

        test_output_dataframe = test_dataframe.copy()
        test_output_dataframe[score_column] = [
            calculate_line_risk_score(
                TOKENIZER(code_line),
                token_risk_scores,
                aggregation_method,
            )
            for code_line in test_output_dataframe['code_line']
        ]

        counterfactual_tokens_found: List[List[str]] = []
        counterfactual_token_counts: List[int] = []

        for code_line in test_output_dataframe['code_line']:
            tokens = TOKENIZER(str(code_line))

            found_tokens = [
                token
                for token in tokens
                if token in counterfactual_token_set
            ]

            counterfactual_tokens_found.append(found_tokens)
            counterfactual_token_counts.append(len(found_tokens))

        test_output_dataframe[
            'line_counterfactual_token_count'
        ] = counterfactual_token_counts
        test_output_dataframe[
            'line_counterfactual_tokens'
        ] = [
            json.dumps(tokens, ensure_ascii=False)
            for tokens in counterfactual_tokens_found
        ]
        test_output_dataframe['line_risk_threshold'] = (
            line_risk_threshold
        )
        test_output_dataframe['line_defect_rate'] = line_defect_rate
        test_output_dataframe['line_score_method'] = aggregation_method
        test_output_dataframe['file_model_name'] = file_model_name
        test_output_dataframe['token_score_type'] = 'suspiciousness_only'

        prediction_path = os.path.join(
            prediction_directory,
            f'{test_release}_clear_{aggregation_method}.csv',
        )
        test_output_dataframe.to_csv(
            prediction_path,
            index=False,
            encoding='utf-8',
        )

        token_path = os.path.join(
            token_directory,
            f'{train_release}_clear_{aggregation_method}_'
            'counterfactual_tokens.json',
        )
        with open(token_path, 'w', encoding='utf-8') as token_file:
            json.dump(
                sorted(counterfactual_token_set),
                token_file,
                indent=2,
                ensure_ascii=False,
            )

        metadata_path = os.path.join(
            token_directory,
            f'{train_release}_clear_{aggregation_method}_metadata.json',
        )
        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(
                {
                    'dataset_name': dataset_name,
                    'train_release': train_release,
                    'test_release': test_release,
                    'data_role_for_threshold_and_tokens': 'complete_previous_release',
                    'file_model_name': file_model_name,
                    'aggregation_method': aggregation_method,
                    'token_score_type': 'suspiciousness_only',
                    'line_risk_threshold': line_risk_threshold,
                    'line_defect_rate': line_defect_rate,
                    'counterfactual_token_count': len(
                        counterfactual_token_set
                    ),
                    'token_intervention': (
                        'Remove one token occurrence from a code line.'
                    ),
                    'test_token_counting': (
                        'Count all matching token occurrences in each '
                        'code line.'
                    ),
                },
                metadata_file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f'[{train_release}] Selected {aggregation_method} '
            'line threshold without confidence '
            f'{line_risk_threshold:.6f} on the complete previous release.'
        )
        print(
            f'[{test_release}] Saved CLEAR {aggregation_method} '
            'line results without confidence to '
            f'{prediction_path}'
        )
        print(
            f'[{train_release}] Saved '
            f'{len(counterfactual_token_set)} line-level '
            'counterfactual tokens without confidence to '
            f'{token_path}'
        )


if __name__ == '__main__':
    datasets_projects = {
        # 'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }

    selected_file_model_name = DEFAULT_FILE_MODEL_NAME

    for dataset_name, projects in datasets_projects.items():
        for project_name, releases in projects.items():
            for release_index in range(1, len(releases)):
                train_release = releases[release_index - 1]
                test_release = releases[release_index]

                run_line_level_counterfactual_analysis(
                    dataset_name=dataset_name,
                    train_release=train_release,
                    test_release=test_release,
                    file_model_name=selected_file_model_name,
                )

                run_line_level_counterfactual_analysis_without_confidence(
                    dataset_name=dataset_name,
                    train_release=train_release,
                    test_release=test_release,
                    file_model_name=selected_file_model_name,
                )

                print(
                    f'[{dataset_name}][{test_release}] Line-level '
                    'counterfactual analysis completed.'
                )
