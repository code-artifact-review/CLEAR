import json
import os
from typing import Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clear.src.my_utils.helper import (
    glance_projects,
    linedp_projects,
)

DATA_DIR = '../Data'
DEFAULT_FILE_MODEL_NAME = 'logistic_regression'
DEFAULT_FILE_DECISION_THRESHOLD = 0.5

FileModelFactory = Callable[[Optional[dict]], BaseEstimator]
FILE_LEVEL_MODEL_FACTORIES: Dict[str, FileModelFactory] = {}


def register_file_level_model(
        model_name: str,
        model_factory: FileModelFactory,
        overwrite: bool = False,
) -> None:

    if not model_name:
        raise ValueError('model_name must not be empty.')

    if model_name in FILE_LEVEL_MODEL_FACTORIES and not overwrite:
        raise ValueError(
            f'File-level model "{model_name}" is already registered.'
        )

    FILE_LEVEL_MODEL_FACTORIES[model_name] = model_factory


def build_logistic_regression_model(
        model_parameters: Optional[dict] = None,
) -> Pipeline:

    parameters = {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'random_state': 42,
    }
    if model_parameters:
        parameters.update(model_parameters)

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(**parameters)),
    ])


register_file_level_model(
    DEFAULT_FILE_MODEL_NAME,
    build_logistic_regression_model,
)


def create_file_level_model(
        model_name: str = DEFAULT_FILE_MODEL_NAME,
        model_parameters: Optional[dict] = None,
) -> BaseEstimator:

    if model_name not in FILE_LEVEL_MODEL_FACTORIES:
        available_models = sorted(FILE_LEVEL_MODEL_FACTORIES)
        raise ValueError(
            f'Unknown file-level model "{model_name}". '
            f'Available models: {available_models}'
        )

    estimator = FILE_LEVEL_MODEL_FACTORIES[model_name](model_parameters)

    if not hasattr(estimator, 'fit'):
        raise TypeError(
            f'The model "{model_name}" does not implement fit().'
        )

    if not (
            hasattr(estimator, 'predict_proba')
            or hasattr(estimator, 'decision_function')
    ):
        raise TypeError(
            f'The model "{model_name}" must implement '
            'predict_proba() or decision_function().'
        )

    return estimator


def predict_bug_probability(
        estimator: BaseEstimator,
        feature_matrix,
) -> np.ndarray:

    if hasattr(estimator, 'predict_proba'):
        probabilities = estimator.predict_proba(feature_matrix)

        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                'File-level predict_proba() must return probabilities '
                'for a binary classifier.'
            )

        return probabilities[:, 1].astype(float)

    decision_values = np.asarray(
        estimator.decision_function(feature_matrix),
        dtype=float,
    )

    if decision_values.ndim != 1:
        raise ValueError(
            'File-level decision_function() must return one value '
            'per file for binary classification.'
        )

    decision_values = np.clip(decision_values, -709, 709)
    return 1.0 / (1.0 + np.exp(-decision_values))


def load_file_token_data(
        dataset_name: str,
        release: str,
) -> pd.DataFrame:

    token_data_path = (
        f'{DATA_DIR}/{dataset_name}/tokens/{release}_tokens.csv'
    )

    if not os.path.exists(token_data_path):
        raise FileNotFoundError(
            f'File-level token data not found: {token_data_path}'
        )

    token_dataframe = pd.read_csv(token_data_path, encoding='utf-8')

    required_columns = {'filename', 'token', 'count', 'Bug'}
    missing_columns = required_columns - set(token_dataframe.columns)
    if missing_columns:
        raise ValueError(
            'Missing required columns in file-level token data: '
            f'{sorted(missing_columns)}'
        )

    if token_dataframe.empty:
        raise ValueError(
            f'File-level token data is empty for release {release}.'
        )

    token_dataframe = token_dataframe.copy()
    token_dataframe['filename'] = token_dataframe['filename'].astype(str)
    token_dataframe['token'] = token_dataframe['token'].astype(str)
    token_dataframe['count'] = pd.to_numeric(
        token_dataframe['count'],
        errors='raise',
    )
    token_dataframe['Bug'] = pd.to_numeric(
        token_dataframe['Bug'],
        errors='raise',
    ).astype(int)

    invalid_labels = set(token_dataframe['Bug'].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            'File-level labels must be binary values 0 or 1. '
            f'Invalid labels: {sorted(invalid_labels)}'
        )

    return token_dataframe


def build_file_feature_table(
        token_dataframe: pd.DataFrame,
        feature_names: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series]:

    file_feature_table = token_dataframe.pivot_table(
        index='filename',
        columns='token',
        values='count',
        aggfunc='sum',
        fill_value=0,
    )

    if feature_names is None:
        feature_names = sorted(
            str(column)
            for column in file_feature_table.columns
        )

    file_feature_table = file_feature_table.reindex(
        columns=feature_names,
        fill_value=0,
    )
    file_feature_table.columns.name = None

    file_labels = (
        token_dataframe.groupby('filename')['Bug']
        .first()
        .reindex(file_feature_table.index)
        .fillna(0)
        .astype(int)
    )

    return file_feature_table, file_labels


def get_file_prediction_result_path(
        dataset_name: str,
        test_release: str,
        model_name: str,
) -> str:

    return os.path.join(
        DATA_DIR,
        dataset_name,
        'results',
        'file_prediction_results',
        model_name,
        f'{test_release}_file_predictions.csv',
    )


def get_validation_file_prediction_result_path(
        dataset_name: str,
        train_release: str,
        model_name: str,
) -> str:

    return os.path.join(
        DATA_DIR,
        dataset_name,
        'results',
        'file_prediction_results',
        model_name,
        f'{train_release}_validation_file_predictions.csv',
    )


def get_file_model_bundle_path(
        dataset_name: str,
        train_release: str,
        test_release: str,
        model_name: str,
) -> str:

    return os.path.join(
        DATA_DIR,
        dataset_name,
        'results',
        'file_prediction_models',
        model_name,
        f'{train_release}_to_{test_release}.joblib',
    )


def save_file_level_model_bundle(
        bundle_path: str,
        estimator: BaseEstimator,
        model_name: str,
        feature_names: list,
        decision_threshold: float,
        dataset_name: str,
        train_release: str,
        test_release: str,
        model_parameters: Optional[dict],
        split_information: dict,
) -> None:

    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)

    model_bundle = {
        'estimator': estimator,
        'model_name': model_name,
        'feature_names': list(feature_names),
        'decision_threshold': float(decision_threshold),
        'dataset_name': dataset_name,
        'train_release': train_release,
        'test_release': test_release,
        'model_parameters': model_parameters or {},
        'split_information': split_information,
    }
    joblib.dump(model_bundle, bundle_path)

    metadata_path = os.path.splitext(bundle_path)[0] + '_metadata.json'
    metadata = {
        key: value
        for key, value in model_bundle.items()
        if key != 'estimator'
    }
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
            ensure_ascii=False,
        )


def load_file_level_model_bundle(bundle_path: str) -> dict:
    """Load a fitted file-level model bundle."""

    if not os.path.exists(bundle_path):
        raise FileNotFoundError(
            f'File-level model bundle not found: {bundle_path}'
        )

    model_bundle = joblib.load(bundle_path)

    required_keys = {
        'estimator',
        'model_name',
        'feature_names',
        'decision_threshold',
        'split_information',
    }
    missing_keys = required_keys - set(model_bundle)
    if missing_keys:
        raise ValueError(
            'Invalid file-level model bundle. Missing keys: '
            f'{sorted(missing_keys)}'
        )

    return model_bundle


def train_and_predict_file_defects(
        dataset_name: str,
        train_release: str,
        test_release: str,
        model_name: str = DEFAULT_FILE_MODEL_NAME,
        model_parameters: Optional[dict] = None,
) -> pd.DataFrame:

    historical_token_dataframe = load_file_token_data(
        dataset_name,
        train_release,
    )
    test_token_dataframe = load_file_token_data(
        dataset_name,
        test_release,
    )

    train_features, train_labels = build_file_feature_table(
        historical_token_dataframe
    )
    feature_names = list(train_features.columns)

    test_features, test_labels = build_file_feature_table(
        test_token_dataframe,
        feature_names=feature_names,
    )

    if train_features.empty:
        raise ValueError(
            f'No training files remain for release {train_release}.'
        )

    if train_labels.nunique() < 2:
        raise ValueError(
            f'Complete previous release {train_release} contains '
            'only one file-level class.'
        )

    estimator = create_file_level_model(
        model_name=model_name,
        model_parameters=model_parameters,
    )
    estimator.fit(train_features, train_labels)

    decision_threshold = DEFAULT_FILE_DECISION_THRESHOLD

    predicted_bug_probabilities = predict_bug_probability(
        estimator,
        test_features,
    )
    predicted_file_labels = (
            predicted_bug_probabilities >= decision_threshold
    ).astype(int)

    prediction_dataframe = pd.DataFrame({
        'filename': test_features.index,
        'predicted_bug_prob': predicted_bug_probabilities,
        'predicted_bug': predicted_file_labels,
        'true_bug': test_labels.values,
        'decision_threshold': decision_threshold,
        'model_name': model_name,
        'data_role': 'test',
    })

    prediction_result_path = get_file_prediction_result_path(
        dataset_name,
        test_release,
        model_name,
    )
    os.makedirs(
        os.path.dirname(prediction_result_path),
        exist_ok=True,
    )
    prediction_dataframe.to_csv(
        prediction_result_path,
        index=False,
        encoding='utf-8',
    )

    training_files = sorted(
        str(file_name)
        for file_name in train_features.index
    )
    split_information = {
        'split_strategy': 'complete_previous_release',
        'training_files': training_files,
        'validation_files': [],
        'training_file_count': len(training_files),
        'validation_file_count': 0,
    }

    model_bundle_path = get_file_model_bundle_path(
        dataset_name,
        train_release,
        test_release,
        model_name,
    )
    save_file_level_model_bundle(
        bundle_path=model_bundle_path,
        estimator=estimator,
        model_name=model_name,
        feature_names=feature_names,
        decision_threshold=decision_threshold,
        dataset_name=dataset_name,
        train_release=train_release,
        test_release=test_release,
        model_parameters=model_parameters,
        split_information=split_information,
    )

    print(
        f'[{train_release}] Trained {model_name} on '
        f'{len(train_features)} files from the complete previous release.'
    )
    print(
        f'[{test_release}] Saved file-level predictions to '
        f'{prediction_result_path}'
    )
    print(
        f'[{test_release}] Saved file-level model bundle to '
        f'{model_bundle_path}'
    )

    return prediction_dataframe


if __name__ == '__main__':
    datasets_projects = {
        #'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }

    selected_model_name = DEFAULT_FILE_MODEL_NAME

    for dataset_name, projects in datasets_projects.items():
        for project_name, releases in projects.items():
            for release_index in range(1, len(releases)):
                train_release = releases[release_index - 1]
                test_release = releases[release_index]

                train_and_predict_file_defects(
                    dataset_name=dataset_name,
                    train_release=train_release,
                    test_release=test_release,
                    model_name=selected_model_name,
                )
