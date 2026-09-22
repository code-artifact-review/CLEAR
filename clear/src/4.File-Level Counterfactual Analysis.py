import json
import os
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from clear.src.my_utils.helper import (
    glance_projects,
    linedp_projects,
)
from file_level_prediction import (
    DEFAULT_FILE_MODEL_NAME,
    build_file_feature_table,
    get_file_model_bundle_path,
    load_file_level_model_bundle,
    load_file_token_data,
    predict_bug_probability,
)

DATA_DIR = '../Data'
DEFAULT_COUNTERFACTUAL_BATCH_SIZE = 512

EXCLUDED_FILE_COUNTERFACTUAL_TOKENS = {
    '<str>',
    'str',
}


def analyze_file_level_counterfactual_tokens(
        dataset_name: str,
        train_release: str,
        test_release: str,
        model_name: str = DEFAULT_FILE_MODEL_NAME,
        counterfactual_batch_size: int = (
                DEFAULT_COUNTERFACTUAL_BATCH_SIZE
        ),
) -> pd.DataFrame:

    if counterfactual_batch_size <= 0:
        raise ValueError(
            'counterfactual_batch_size must be greater than zero.'
        )

    model_bundle_path = get_file_model_bundle_path(
        dataset_name=dataset_name,
        train_release=train_release,
        test_release=test_release,
        model_name=model_name,
    )
    model_bundle = load_file_level_model_bundle(
        model_bundle_path
    )

    estimator = model_bundle['estimator']
    feature_names = list(model_bundle['feature_names'])
    decision_threshold = float(
        model_bundle['decision_threshold']
    )
    historical_token_dataframe = load_file_token_data(
        dataset_name,
        train_release,
    )

    historical_feature_table, historical_labels = (
        build_file_feature_table(
            historical_token_dataframe,
            feature_names=feature_names,
        )
    )

    if historical_feature_table.empty:
        raise ValueError(
            f'No historical files remain for release {train_release}.'
        )

    original_bug_probabilities = predict_bug_probability(
        estimator,
        historical_feature_table,
    )

    summary_rows = []
    detail_rows = []
    global_counterfactual_token_frequency = Counter()

    file_iterator = zip(
        historical_feature_table.index,
        historical_feature_table.to_numpy(dtype=float),
        original_bug_probabilities,
        historical_labels.reindex(
            historical_feature_table.index
        ).values,
    )

    for (
            file_name,
            original_feature_vector,
            original_bug_probability,
            true_bug_label,
    ) in tqdm(
        file_iterator,
        total=len(historical_feature_table),
        desc=(
                f'{train_release} -  '
                'file-level counterfactual token analysis'
        ),
        unit='file',
        dynamic_ncols=True,
        leave=False,
    ):
        predicted_bug = int(
            original_bug_probability >= decision_threshold
        )

        if int(true_bug_label) != 1 or predicted_bug == 0:
            continue

        present_token_indices = np.flatnonzero(
            original_feature_vector > 0
        )

        file_counterfactual_tokens: List[str] = []
        file_probability_drops: List[float] = []

        for batch_start in range(
                0,
                len(present_token_indices),
                counterfactual_batch_size,
        ):
            batch_token_indices = present_token_indices[
                                  batch_start:
                                  batch_start + counterfactual_batch_size
                                  ]

            counterfactual_feature_matrix = np.repeat(
                original_feature_vector.reshape(1, -1),
                repeats=len(batch_token_indices),
                axis=0,
            )
            counterfactual_feature_matrix[
                np.arange(len(batch_token_indices)),
                batch_token_indices,
            ] = 0.0

            counterfactual_feature_dataframe = pd.DataFrame(
                counterfactual_feature_matrix,
                columns=feature_names,
            )
            counterfactual_bug_probabilities = (
                predict_bug_probability(
                    estimator,
                    counterfactual_feature_dataframe,
                )
            )

            for (
                    token_index,
                    counterfactual_bug_probability,
            ) in zip(
                batch_token_indices,
                counterfactual_bug_probabilities,
            ):
                token = feature_names[int(token_index)]

                if token in EXCLUDED_FILE_COUNTERFACTUAL_TOKENS:
                    continue

                original_token_count = float(
                    original_feature_vector[int(token_index)]
                )
                probability_drop = float(
                    original_bug_probability
                    - counterfactual_bug_probability
                )
                changes_prediction = (
                        counterfactual_bug_probability
                        < decision_threshold
                )

                if not changes_prediction:
                    continue

                file_counterfactual_tokens.append(token)
                file_probability_drops.append(probability_drop)
                global_counterfactual_token_frequency[token] += 1

                detail_rows.append({
                    'filename': file_name,
                    'token': token,
                    'original_token_count': original_token_count,
                    'original_bug_probability': float(
                        original_bug_probability
                    ),
                    'counterfactual_bug_probability': float(
                        counterfactual_bug_probability
                    ),
                    'probability_drop': probability_drop,
                    'decision_threshold': decision_threshold,
                    'model_name': model_name,
                    'true_bug': int(true_bug_label),
                })

        summary_rows.append({
            'filename': file_name,
            'original_bug_probability': float(
                original_bug_probability
            ),
            'predicted_bug': predicted_bug,
            'true_bug': int(true_bug_label),
            'decision_threshold': decision_threshold,
            'file_counterfactual_token_count': len(
                file_counterfactual_tokens
            ),
            'file_counterfactual_tokens': json.dumps(
                file_counterfactual_tokens,
                ensure_ascii=False,
            ),
            'maximum_probability_drop': (
                max(file_probability_drops)
                if file_probability_drops
                else 0.0
            ),
            'mean_probability_drop': (
                float(np.mean(file_probability_drops))
                if file_probability_drops
                else 0.0
            ),
            'model_name': model_name,
        })

    output_directory = os.path.join(
        DATA_DIR,
        dataset_name,
        'results',
        'file_counterfactual_analysis',
        model_name,
    )
    os.makedirs(output_directory, exist_ok=True)

    summary_columns = [
        'filename',
        'original_bug_probability',
        'predicted_bug',
        'true_bug',
        'decision_threshold',
        'file_counterfactual_token_count',
        'file_counterfactual_tokens',
        'maximum_probability_drop',
        'mean_probability_drop',
        'model_name',
    ]
    detail_columns = [
        'filename',
        'token',
        'original_token_count',
        'original_bug_probability',
        'counterfactual_bug_probability',
        'probability_drop',
        'decision_threshold',
        'model_name',
        'true_bug',
    ]

    summary_dataframe = pd.DataFrame(
        summary_rows,
        columns=summary_columns,
    )
    detail_dataframe = pd.DataFrame(
        detail_rows,
        columns=detail_columns,
    )

    output_prefix = f'{train_release}_full_previous_release'
    legacy_output_prefix = f'{train_release}_validation'
    summary_path = os.path.join(
        output_directory,
        f'{output_prefix}_file_counterfactual_summary.csv',
    )
    detail_path = os.path.join(
        output_directory,
        f'{output_prefix}_file_counterfactual_tokens.csv',
    )
    frequency_path = os.path.join(
        output_directory,
        f'{output_prefix}_file_counterfactual_token_frequency.json',
    )
    metadata_path = os.path.join(
        output_directory,
        f'{output_prefix}_file_counterfactual_metadata.json',
    )

    legacy_summary_path = os.path.join(
        output_directory,
        f'{legacy_output_prefix}_file_counterfactual_summary.csv',
    )
    legacy_detail_path = os.path.join(
        output_directory,
        f'{legacy_output_prefix}_file_counterfactual_tokens.csv',
    )
    legacy_frequency_path = os.path.join(
        output_directory,
        f'{legacy_output_prefix}_file_counterfactual_token_frequency.json',
    )
    legacy_metadata_path = os.path.join(
        output_directory,
        f'{legacy_output_prefix}_file_counterfactual_metadata.json',
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
        encoding='utf-8',
    )
    detail_dataframe.to_csv(
        detail_path,
        index=False,
        encoding='utf-8',
    )

    summary_dataframe.to_csv(
        legacy_summary_path,
        index=False,
        encoding='utf-8',
    )
    detail_dataframe.to_csv(
        legacy_detail_path,
        index=False,
        encoding='utf-8',
    )

    sorted_frequency_dictionary = dict(
        sorted(
            global_counterfactual_token_frequency.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )

    with open(
            frequency_path,
            'w',
            encoding='utf-8',
    ) as frequency_file:
        json.dump(
            sorted_frequency_dictionary,
            frequency_file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
            legacy_frequency_path,
            'w',
            encoding='utf-8',
    ) as legacy_frequency_file:
        json.dump(
            sorted_frequency_dictionary,
            legacy_frequency_file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
            metadata_path,
            'w',
            encoding='utf-8',
    ) as metadata_file:
        json.dump(
            {
                'dataset_name': dataset_name,
                'train_release': train_release,
                'test_release': test_release,
                'data_role': 'complete_previous_release',
                'model_name': model_name,
                'decision_threshold': decision_threshold,
                'analyzed_correctly_predicted_defective_files': len(
                    summary_dataframe
                ),
                'identified_file_counterfactual_tokens': len(
                    detail_dataframe
                ),
                'unique_file_counterfactual_tokens': len(
                    global_counterfactual_token_frequency
                ),
                'counterfactual_intervention': (
                    'Set the selected token count to zero in the '
                    'file-level BoT vector.'
                ),
                'counterfactual_criterion': (
                    'The complete-previous-release file is truly '
                    'defective and its predicted defect probability '
                    'changes from not lower than the fixed threshold '
                    'to lower than the threshold.'
                ),
                'training_data_reuse': (
                    'The same complete previous release is used for '
                    'file-level model fitting and counterfactual analysis.'
                ),
            },
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    with open(
            metadata_path,
            'r',
            encoding='utf-8',
    ) as metadata_file:
        metadata = json.load(metadata_file)

    metadata['compatibility_output_name'] = True
    metadata['canonical_metadata_path'] = metadata_path

    with open(
            legacy_metadata_path,
            'w',
            encoding='utf-8',
    ) as legacy_metadata_file:
        json.dump(
            metadata,
            legacy_metadata_file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f'[{train_release}] Saved '
        f'file-level counterfactual summary to {summary_path}'
    )
    print(
        f'[{train_release}] Saved '
        f'file-level counterfactual tokens to {detail_path}'
    )

    return summary_dataframe


if __name__ == '__main__':
    datasets_projects = {
        # 'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }

    selected_model_name = DEFAULT_FILE_MODEL_NAME

    for dataset_name, projects in datasets_projects.items():
        for project_name, releases in projects.items():
            for release_index in range(1, len(releases)):
                train_release = releases[release_index - 1]
                test_release = releases[release_index]

                analyze_file_level_counterfactual_tokens(
                    dataset_name=dataset_name,
                    train_release=train_release,
                    test_release=test_release,
                    model_name=selected_model_name,
                )

                print(
                    f'[{train_release}] Complete '
                    'file-level counterfactual analysis completed.'
                )
