import json
import math
import os
from operator import itemgetter

import pandas as pd

from baselines.sound.src.utils.config import (
    GLANCE_PROJECT_RELEASE_LIST,
    LINEDP_PROJECT_RELEASE_LIST,
)
from utils.helper import read_file_level_dataset

glance_releases = GLANCE_PROJECT_RELEASE_LIST
linedp_releases = LINEDP_PROJECT_RELEASE_LIST

DATASET_CONFIGS = {
    'glance_dataset': {
        'releases': glance_releases,
        'data_root': '../Data/glance_dataset',
        'file_level_path': (
            '../../../dataset/'
            'glance_dataset/File-level/'
        ),
        'line_level_path': (
            '../../../dataset/'
            'glance_dataset/Line-level/'
        ),
    },
    'linedp_dataset': {
        'releases': linedp_releases,
        'data_root': '../Data/linedp_dataset',
        'file_level_path': (
            '../../../dataset/'
            'linedp_dataset/File-level/'
        ),
        'line_level_path': (
            '../../../dataset/'
            'linedp_dataset/Line-level/'
        ),
    },
}


def get_tokens_split(tokens_file_name) -> set:
    tokens_csv = pd.read_csv(
        tokens_file_name,
        encoding='ISO-8859-1',
    )
    tokens_set = set(tokens_csv.columns)
    return tokens_set


def tarantula(
        n_cf,
        n_cs,
        num_of_bug_lines,
        num_of_lines,
):
    if (
            n_cf / num_of_bug_lines
            + n_cs
            / (
            num_of_lines
            - num_of_bug_lines
    )
            == 0
    ):
        return 0

    return (
            n_cf / num_of_bug_lines
    ) / (
            n_cf / num_of_bug_lines
            + n_cs
            / (
                    num_of_lines
                    - num_of_bug_lines
            )
    )


def ochiai(
        n_cf,
        n_cs,
        num_of_bug_lines,
):
    if (
            num_of_bug_lines
            * (n_cf + n_cs)
            == 0
    ):
        return 0

    return n_cf / math.sqrt(
        num_of_bug_lines
        * (n_cf + n_cs)
    )


def op2(
        n_cf,
        n_cs,
        num_of_bug_lines,
        num_of_lines,
):
    if (
            num_of_lines
            - num_of_bug_lines
            + 1
            == 0
    ):
        return 0

    return (
            n_cf
            - n_cs
            / (
                    num_of_lines
                    - num_of_bug_lines
                    + 1
            )
    )


def barinel(
        n_cf,
        n_cs,
):
    if n_cs + n_cf == 0:
        return 0

    return 1 - n_cs / (n_cs + n_cf)


def dstar(
        n_cf,
        n_cs,
        num_of_bug_lines,
):
    if (
            n_cs
            + num_of_bug_lines
            - n_cf
            == 0
    ):
        return 0

    return (
            n_cf * n_cf
    ) / (
            n_cs
            + num_of_bug_lines
            - n_cf
    )


def read_line_level_dataset_from_path(
        release,
        file_path,
):
    """
    Read one line-level dataset from an explicit directory.

    This preserves the original SOUND parsing behavior while allowing
    GLANCE and LINE-DP to use isolated dataset directories.
    """

    if release == '':
        return {}

    path = os.path.join(
        file_path,
        release
        + '_defective_lines_dataset.csv',
    )

    with open(
            path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as file:
        lines = file.readlines()

    file_buggy_lines_dict = {}

    for line in lines[1:]:
        values = line.split(',', 2)
        file_name = values[0]
        buggy_line_number = int(values[1])

        if file_name not in file_buggy_lines_dict:
            file_buggy_lines_dict[
                file_name
            ] = [buggy_line_number]
        else:
            file_buggy_lines_dict[
                file_name
            ].append(
                buggy_line_number
            )

    return file_buggy_lines_dict


def get_oracle_lines(
        release,
        line_level_data_path,
):
    oracle_line_dict = (
        read_line_level_dataset_from_path(
            release,
            line_level_data_path,
        )
    )
    oracle_line_list = set()

    for file_name in oracle_line_dict:
        oracle_line_list.update(
            [
                f'{file_name}:{line}'
                for line
                in oracle_line_dict[file_name]
            ]
        )

    return (
        oracle_line_dict,
        oracle_line_list,
    )


def normalize_score(score_dict: dict):
    max_value = max(score_dict.values())
    min_value = min(score_dict.values())

    normalized_dict = {
        key: (
                     value - min_value
             ) / (
                     max_value - min_value
             )
        for key, value
        in score_dict.items()
    }

    return normalized_dict


def get_dataset_paths(
        dataset_name,
        release,
):
    config = DATASET_CONFIGS[
        dataset_name
    ]
    data_root = config['data_root']

    n_cf_file_name = os.path.join(
        data_root,
        'n_score',
        release + '_cf.json',
    )
    n_cs_file_name = os.path.join(
        data_root,
        'n_score',
        release + '_cs.json',
    )
    tokens_file_name = os.path.join(
        data_root,
        'tokens',
        release + '_tokens.csv',
    )
    score_directory = os.path.join(
        data_root,
        'score',
    )

    return (
        config,
        n_cf_file_name,
        n_cs_file_name,
        tokens_file_name,
        score_directory,
    )


def load_n_score_inputs(
        dataset_name,
        release,
):
    (
        config,
        n_cf_file_name,
        n_cs_file_name,
        tokens_file_name,
        score_directory,
    ) = get_dataset_paths(
        dataset_name,
        release,
    )

    with open(
            n_cf_file_name,
            'r',
    ) as file:
        n_cf = json.load(file)

    with open(
            n_cs_file_name,
            'r',
    ) as file:
        n_cs = json.load(file)

    tokens = get_tokens_split(
        tokens_file_name
    )

    os.makedirs(
        score_directory,
        exist_ok=True,
    )

    return (
        config,
        n_cf,
        n_cs,
        tokens,
        score_directory,
    )


def save_normalized_score(
        score,
        output_file_name,
):
    normal_score = normalize_score(score)
    sorted_dict = dict(
        sorted(
            normal_score.items(),
            key=itemgetter(1),
            reverse=True,
        )
    )

    with open(
            output_file_name,
            'w',
    ) as file:
        json.dump(
            sorted_dict,
            file,
        )


def evaluate_tarantula_score(
        dataset_name,
        release,
):
    (
        config,
        n_cf,
        n_cs,
        tokens,
        score_directory,
    ) = load_n_score_inputs(
        dataset_name,
        release,
    )

    (
        _,
        test_text_lines,
        _,
        _,
        _,
    ) = read_file_level_dataset(
        release,
        file_path=config[
            'file_level_path'
        ],
    )
    num_of_lines = sum(
        len(lines)
        for lines in test_text_lines
    )

    (
        _,
        oracle_line_set,
    ) = get_oracle_lines(
        release,
        config['line_level_path'],
    )
    num_of_bug_lines = len(
        oracle_line_set
    )

    score = {}

    for token in tokens:
        score[token] = tarantula(
            n_cf[token],
            n_cs[token],
            num_of_bug_lines,
            num_of_lines,
        )

    output_file_name = os.path.join(
        score_directory,
        release
        + '_tarantula_normal.json',
    )
    save_normalized_score(
        score,
        output_file_name,
    )


def evaluate_ochiai_score(
        dataset_name,
        release,
):
    (
        config,
        n_cf,
        n_cs,
        tokens,
        score_directory,
    ) = load_n_score_inputs(
        dataset_name,
        release,
    )

    (
        _,
        oracle_line_set,
    ) = get_oracle_lines(
        release,
        config['line_level_path'],
    )
    num_of_bug_lines = len(
        oracle_line_set
    )

    score = {}

    for token in tokens:
        score[token] = ochiai(
            n_cf[token],
            n_cs[token],
            num_of_bug_lines,
        )

    output_file_name = os.path.join(
        score_directory,
        release
        + '_ochiai_normal.json',
    )
    save_normalized_score(
        score,
        output_file_name,
    )


def evaluate_op2_score(
        dataset_name,
        release,
):
    (
        config,
        n_cf,
        n_cs,
        tokens,
        score_directory,
    ) = load_n_score_inputs(
        dataset_name,
        release,
    )

    (
        _,
        test_text_lines,
        _,
        _,
        _,
    ) = read_file_level_dataset(
        release,
        file_path=config[
            'file_level_path'
        ],
    )
    num_of_lines = sum(
        len(lines)
        for lines in test_text_lines
    )

    (
        _,
        oracle_line_set,
    ) = get_oracle_lines(
        release,
        config['line_level_path'],
    )
    num_of_bug_lines = len(
        oracle_line_set
    )

    score = {}

    for token in tokens:
        score[token] = op2(
            n_cf[token],
            n_cs[token],
            num_of_bug_lines,
            num_of_lines,
        )

    output_file_name = os.path.join(
        score_directory,
        release
        + '_op2_normal.json',
    )
    save_normalized_score(
        score,
        output_file_name,
    )


def evaluate_barinel_score(
        dataset_name,
        release,
):
    (
        _,
        n_cf,
        n_cs,
        tokens,
        score_directory,
    ) = load_n_score_inputs(
        dataset_name,
        release,
    )

    score = {}

    for token in tokens:
        score[token] = barinel(
            n_cf[token],
            n_cs[token],
        )

    output_file_name = os.path.join(
        score_directory,
        release
        + '_barinel_normal.json',
    )
    save_normalized_score(
        score,
        output_file_name,
    )


def evaluate_dstar_score(
        dataset_name,
        release,
):
    (
        config,
        n_cf,
        n_cs,
        tokens,
        score_directory,
    ) = load_n_score_inputs(
        dataset_name,
        release,
    )

    (
        _,
        oracle_line_set,
    ) = get_oracle_lines(
        release,
        config['line_level_path'],
    )
    num_of_bug_lines = len(
        oracle_line_set
    )

    score = {}

    for token in tokens:
        score[token] = dstar(
            n_cf[token],
            n_cs[token],
            num_of_bug_lines,
        )

    output_file_name = os.path.join(
        score_directory,
        release
        + '_dstar_normal.json',
    )
    save_normalized_score(
        score,
        output_file_name,
    )


def evaluate_release(
        dataset_name,
        release,
):
    print(
        f'[{dataset_name}] '
        f'Evaluating {release}'
    )

    evaluate_tarantula_score(
        dataset_name,
        release,
    )
    evaluate_ochiai_score(
        dataset_name,
        release,
    )
    evaluate_op2_score(
        dataset_name,
        release,
    )
    evaluate_barinel_score(
        dataset_name,
        release,
    )
    evaluate_dstar_score(
        dataset_name,
        release,
    )

    print(
        f'[{dataset_name}] '
        f'Finished {release}'
    )


def evaluate_dataset(dataset_name):
    releases = DATASET_CONFIGS[
        dataset_name
    ]['releases']

    for release in releases:
        evaluate_release(
            dataset_name,
            release,
        )


if __name__ == '__main__':
    # evaluate_dataset(
    #     'glance_dataset'
    # )
    evaluate_dataset(
        'linedp_dataset'
    )
