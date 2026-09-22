import os
import re
import warnings

import numpy as np
import pandas as pd

from clear.src.my_utils.helper import (
    read_line_level_dataset,
    read_file_level_dataset,
    glance_projects,
    linedp_projects,
)

warnings.simplefilter(
    action='ignore',
    category=pd.errors.PerformanceWarning,
)

dataset_dir = '../../dataset'

JAVA_STRING_PATTERN = re.compile(
    r'""(?:\\.|[^"\\])*""'
    r'|'
    r'"(?:\\.|[^"\\])*"'
)


def preprocess_code_line(code_line):
    code_line = JAVA_STRING_PATTERN.sub('<str>', code_line)
    return code_line.strip()


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

    df['filename'] = [filename] * len(code_lines)
    df['is_test_file'] = [is_test] * len(code_lines)
    df['code_line'] = preprocess_code_lines
    df['line_number'] = np.arange(
        1,
        len(code_lines) + 1,
    )
    df['is_comment'] = is_comments
    df['is_blank'] = is_blank_line

    return df


def preprocess_data(which_dataset, rel):
    file_lvl_dir = (
        f'{dataset_dir}/{which_dataset}/File-level/'
    )
    line_lvl_dir = (
        f'{dataset_dir}/{which_dataset}/Line-level/'
    )
    save_dir = (
        f'../Data/{which_dataset}/preprocessed_data/'
    )

    os.makedirs(save_dir, exist_ok=True)

    file_path = (
            file_lvl_dir
            + rel
            + '_ground-truth-files_dataset.csv'
    )
    line_path = (
            line_lvl_dir
            + rel
            + '_defective_lines_dataset.csv'
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f'File-level dataset not found: {file_path}'
        )

    if not os.path.exists(line_path):
        raise FileNotFoundError(
            f'Line-level dataset not found: {line_path}'
        )

    texts, texts_lines, numeric_labels, src_files, texts_lines_without_comments = (
        read_file_level_dataset(
            which_dataset,
            rel,
        )
    )
    file_buggy_lines_dict = read_line_level_dataset(
        which_dataset,
        rel,
    )

    file_level_data = pd.DataFrame()
    file_level_data['File'] = src_files
    file_level_data['Bug'] = numeric_labels
    file_level_data['Bug'] = (
        file_level_data['Bug'].astype(bool)
    )
    file_level_data['SRC'] = [
        '\n'.join(lines)
        for lines in texts_lines
    ]

    rows = []
    for file, lines in file_buggy_lines_dict.items():
        for line in lines:
            rows.append({
                'File': file,
                'Line_number': line,
            })

    line_level_data = pd.DataFrame(
        rows,
        columns=[
            'File',
            'Line_number',
        ],
    )

    file_level_data = file_level_data.fillna('')

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
                    line_level_data['File'] == filename
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

    all_df = pd.concat(preprocessed_df_list)
    all_df = all_df[
        (
                (all_df['is_comment'] == False)
                & (all_df['is_blank'] == False)
        )
        |
        (
                (all_df['is_comment'] == 'False')
                & (all_df['is_blank'] == 'False')
        )
        ]
    all_df['original_line_number'] = all_df['line_number']
    all_df['line_number'] = (
            all_df.groupby('filename').cumcount() + 1
    )
    all_df.to_csv(
        save_dir + rel + '.csv',
        index=False,
        encoding='utf-8',
    )
    print('finish release {}'.format(rel))


if __name__ == '__main__':
    datasets_projects = {
        # 'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }

    for which_dataset, projects in datasets_projects.items():
        for project, releases in projects.items():
            for release in releases:
                preprocess_data(
                    which_dataset,
                    release,
                )
