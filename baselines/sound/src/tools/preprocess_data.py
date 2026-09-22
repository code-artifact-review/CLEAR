import os
import re
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from baselines.sound.src.utils.config_for_deeplinedp import (
    all_releases as glance_all_releases,
)
from clear.src.my_utils.helper import linedp_projects


datasets_projects = {
    # 'glance_dataset': glance_all_releases,
    'linedp_dataset': linedp_projects,
}

char_to_remove = [
    '+',
    '-',
    '*',
    '/',
    '=',
    '++',
    '--',
    '\\',
    '<str>',
    '<char>',
    '|',
    '&',
    '!',
]


def is_comment_line(code_line, comments_list):
    code_line = code_line.strip()

    if len(code_line) == 0:
        return False
    elif code_line.startswith('//'):
        return True
    elif code_line in comments_list:
        return True

    return False


def is_empty_line(code_line):
    if len(code_line.strip()) == 0:
        return True

    return False


def preprocess_code_line(code_line):
    code_line = re.sub("\'\'", "\'", code_line)
    code_line = re.sub(
        "\".*?\"",
        "<str>",
        code_line,
    )
    code_line = re.sub(
        "\'.*?\'",
        "<char>",
        code_line,
    )
    code_line = re.sub(
        '\b\d+\b',
        '',
        code_line,
    )
    code_line = re.sub(
        "\\[.*?\\]",
        '',
        code_line,
    )
    code_line = re.sub(
        "[\\.|,|:|;|{|}|(|)]",
        ' ',
        code_line,
    )

    for char in char_to_remove:
        code_line = code_line.replace(
            char,
            ' ',
        )

    code_line = code_line.strip()

    return code_line


def create_code_df(code_str, filename):
    df = pd.DataFrame()

    code_lines = code_str.splitlines()

    preprocess_code_lines = []
    is_comments = []
    is_blank_line = []

    comments = re.findall(
        r'(/\*[\s\S]*?\*/)',
        code_str,
        re.DOTALL,
    )
    comments_str = '\n'.join(comments)
    comments_list = comments_str.split('\n')

    for line in code_lines:
        line = line.strip()
        is_comment = is_comment_line(
            line,
            comments_list,
        )
        is_comments.append(is_comment)

        if not is_comment:
            line = preprocess_code_line(line)

        is_blank_line.append(
            is_empty_line(line)
        )
        preprocess_code_lines.append(line)

    if 'test' in filename:
        is_test = True
    else:
        is_test = False

    df['filename'] = (
        [filename] * len(code_lines)
    )
    df['is_test_file'] = (
        [is_test] * len(code_lines)
    )
    df['code_line'] = preprocess_code_lines
    df['line_number'] = np.arange(
        1,
        len(code_lines) + 1,
    )
    df['is_comment'] = is_comments
    df['is_blank'] = is_blank_line

    return df


def preprocess_data(
        dataset_name,
        project_name,
        all_releases,
):
    data_root_dir = (
        f'../../../../dataset/{dataset_name}/'
        'processed/'
    )
    save_dir = (
        f'../../../../dataset/{dataset_name}/'
        'preprocessed_data/'
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_level_dir = (
        data_root_dir + 'File-level/'
    )
    line_level_dir = (
        data_root_dir + 'Line-level/'
    )

    current_releases = all_releases[
        project_name
    ]

    for release in current_releases:
        file_path = (
            file_level_dir
            + release
            + '_ground-truth-files_dataset.csv'
        )
        line_path = (
            line_level_dir
            + release
            + '_defective_lines_dataset.csv'
        )

        print(file_path)

        file_level_data = pd.read_csv(
            file_path
        )
        line_level_data = pd.read_csv(
            line_path
        )

        file_level_data = (
            file_level_data.fillna('')
        )

        buggy_files = list(
            line_level_data['File'].unique()
        )

        preprocessed_df_list = []

        for idx, row in file_level_data.iterrows():
            filename = row['File']

            if '.java' not in filename:
                continue

            code = row['SRC']
            label = row['Bug']

            code_df = create_code_df(
                code,
                filename,
            )
            code_df['file-label'] = (
                [label] * len(code_df)
            )
            code_df['line-label'] = (
                [False] * len(code_df)
            )

            if filename in buggy_files:
                buggy_lines = list(
                    line_level_data[
                        line_level_data['File']
                        == filename
                    ]['Line_number']
                )
                code_df['line-label'] = (
                    code_df['line_number'].isin(
                        buggy_lines
                    )
                )

            if len(code_df) > 0:
                preprocessed_df_list.append(
                    code_df
                )

        all_df = pd.concat(
            preprocessed_df_list
        )
        all_df.to_csv(
            save_dir + release + '.csv',
            index=False,
        )

        print(
            f'[{dataset_name}] '
            f'finish release {release}'
        )


for dataset_name, projects in datasets_projects.items():
    for project_name in projects:
        preprocess_data(
            dataset_name,
            project_name,
            projects,
        )
