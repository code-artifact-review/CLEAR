import json
import os
from collections import Counter
from math import log
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from clear.src.my_utils.helper import (
    glance_projects,
    linedp_projects,
)

DATA_ROOT = '../Data'
TOKEN_VECTORIZER = CountVectorizer(lowercase=False)
TOKENIZER = TOKEN_VECTORIZER.build_tokenizer()


def load_preprocessed_line_data(
        dataset_name: str,
        release: str,
) -> pd.DataFrame:

    input_path = (
        f'{DATA_ROOT}/{dataset_name}/preprocessed_data/'
        f'{release}.csv'
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'Preprocessed data file not found: {input_path}'
        )

    line_dataframe = pd.read_csv(input_path, encoding='utf-8')

    required_columns = {'filename', 'code_line', 'line-label'}
    missing_columns = required_columns - set(line_dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required columns in preprocessed data: '
            f'{sorted(missing_columns)}'
        )

    if line_dataframe.empty:
        raise ValueError(
            f'Preprocessed data is empty for release {release}.'
        )

    line_dataframe = line_dataframe.copy()
    line_dataframe['filename'] = (
        line_dataframe['filename'].astype(str)
    )
    line_dataframe['code_line'] = (
        line_dataframe['code_line'].fillna('').astype(str)
    )
    line_dataframe['line-label'] = (
        pd.to_numeric(
            line_dataframe['line-label'],
            errors='raise',
        ).astype(int)
    )

    invalid_labels = set(
        line_dataframe['line-label'].unique()
    ) - {0, 1}
    if invalid_labels:
        raise ValueError(
            'Line-level labels must be binary values 0 or 1. '
            f'Invalid labels: {sorted(invalid_labels)}'
        )

    return line_dataframe


def extract_unique_tokens(code_line: str) -> set:

    return set(TOKENIZER(code_line))


def calculate_token_score_components(
        line_dataframe: pd.DataFrame,
) -> tuple:


    defective_line_count_by_token = Counter()
    clean_line_count_by_token = Counter()

    for line_label, code_line in zip(
            line_dataframe['line-label'],
            line_dataframe['code_line'],
    ):
        unique_tokens = extract_unique_tokens(code_line)

        if line_label == 1:
            defective_line_count_by_token.update(unique_tokens)
        else:
            clean_line_count_by_token.update(unique_tokens)

    vocabulary = (
            set(defective_line_count_by_token)
            | set(clean_line_count_by_token)
    )

    suspiciousness_scores = {}
    confidence_scores = {}
    token_scores = {}

    for token in vocabulary:
        defective_line_count = defective_line_count_by_token[token]
        clean_line_count = clean_line_count_by_token[token]
        total_line_count = (
                defective_line_count + clean_line_count
        )

        if total_line_count == 0:
            continue

        suspiciousness_score = (
                defective_line_count / total_line_count
        )
        confidence_score = log(1 + defective_line_count)
        confidence_enhanced_score = (
                suspiciousness_score * confidence_score
        )

        suspiciousness_scores[token] = (
            suspiciousness_score
        )
        confidence_scores[token] = (
            confidence_score
        )
        token_scores[token] = (
            confidence_enhanced_score
        )

    return (
        suspiciousness_scores,
        confidence_scores,
        token_scores,
    )


def calculate_confidence_enhanced_token_scores(
        dataset_name: str,
        release: str,
) -> dict:

    line_dataframe = load_preprocessed_line_data(
        dataset_name,
        release,
    )
    if line_dataframe.empty:
        raise ValueError(
            f'No code lines remain for release {release}.'
        )

    output_path = (
        f'{DATA_ROOT}/{dataset_name}/token_scores/'
        f'{release}_confidence_enhanced_token_scores.json'
    )
    suspiciousness_output_path = (
        f'{DATA_ROOT}/{dataset_name}/token_scores/'
        f'{release}_suspiciousness_token_scores.json'
    )
    confidence_output_path = (
        f'{DATA_ROOT}/{dataset_name}/token_scores/'
        f'{release}_confidence_token_scores.json'
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    (
        final_suspiciousness_scores,
        final_confidence_scores,
        final_token_risk_scores,
    ) = calculate_token_score_components(
        line_dataframe
    )

    final_token_risk_scores = dict(
        sorted(
            final_token_risk_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    final_suspiciousness_scores = dict(
        sorted(
            final_suspiciousness_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )
    final_confidence_scores = dict(
        sorted(
            final_confidence_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(
            final_token_risk_scores,
            output_file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
            suspiciousness_output_path,
            'w',
            encoding='utf-8',
    ) as suspiciousness_output_file:
        json.dump(
            final_suspiciousness_scores,
            suspiciousness_output_file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
            confidence_output_path,
            'w',
            encoding='utf-8',
    ) as confidence_output_file:
        json.dump(
            final_confidence_scores,
            confidence_output_file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f'[{dataset_name}][{release}] Saved '
        f'{len(final_token_risk_scores)} '
        f'confidence-enhanced token scores to {output_path}'
    )
    print(
        f'[{dataset_name}][{release}] Saved '
        f'{len(final_suspiciousness_scores)} '
        f'suspiciousness token scores to '
        f'{suspiciousness_output_path}'
    )
    print(
        f'[{dataset_name}][{release}] Saved '
        f'{len(final_confidence_scores)} '
        f'confidence token scores to {confidence_output_path}'
    )

    return final_token_risk_scores


if __name__ == '__main__':
    datasets_projects = {
        # 'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }

    for dataset_name, projects in datasets_projects.items():
        for project_name, releases in projects.items():
            for release in releases:
                calculate_confidence_enhanced_token_scores(
                    dataset_name=dataset_name,
                    release=release,
                )
