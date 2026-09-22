import os
import re

import pandas as pd
from tqdm import tqdm

from baselines.sound.src.utils.config_for_deeplinedp import (
    all_eval_releases as glance_all_eval_releases,
    all_releases as glance_all_releases,
    all_train_releases as glance_all_train_releases,
)
from clear.src.my_utils.helper import linedp_projects

linedp_all_train_releases = {
    project_name: releases[0]
    for project_name, releases in linedp_projects.items()
}
linedp_all_eval_releases = {
    project_name: releases[1:]
    for project_name, releases in linedp_projects.items()
}

datasets_release_config = {
    'glance_dataset': {
        'all_train_releases': glance_all_train_releases,
        'all_eval_releases': glance_all_eval_releases,
        'all_releases': glance_all_releases,
    },
    'linedp_dataset': {
        'all_train_releases': linedp_all_train_releases,
        'all_eval_releases': linedp_all_eval_releases,
        'all_releases': linedp_projects,
    },
}


def export_df_to_files(
        data_df,
        code_file_dir,
        line_file_dir,
):
    for filename, df in tqdm(
            data_df.groupby('filename')
    ):
        code_lines = list(df['code_line'])
        code_str = '\n'.join(code_lines)
        code_str = code_str.lower()
        line_num = list(df['line_number'])
        line_num = [
            str(line)
            for line in line_num
        ]

        code_filename = (
                filename
                .replace('/', '_')
                .replace('.java', '')
                + '.txt'
        )
        line_filename = (
                filename
                .replace('/', '_')
                .replace('.java', '')
                + '_line_num.txt'
        )

        with open(
                code_file_dir + code_filename,
                'w',
                encoding='utf-8',
        ) as file:
            file.write(code_str)

        with open(
                line_file_dir + line_filename,
                'w',
        ) as line_file:
            line_file.write('\n'.join(line_num))


def export_ngram_data_each_release(
        dataset_name,
        release,
        is_train=False,
):
    base_data_dir = (
        f'../../../../dataset/{dataset_name}/'
        'preprocessed_data/'
    )
    data_for_ngram_dir = (
        f'../../../../dataset/{dataset_name}/'
        'n_gram_data/'
    )

    file_dir = data_for_ngram_dir + release + '/'
    file_src_dir = file_dir + 'src/'
    file_line_num_dir = file_dir + 'line_num/'

    if not os.path.exists(file_src_dir):
        os.makedirs(file_src_dir)

    if not os.path.exists(file_line_num_dir):
        os.makedirs(file_line_num_dir)

    data_df = pd.read_csv(
        base_data_dir + release + '.csv',
        encoding='latin',
    )

    if is_train:
        data_df = data_df[
            (data_df['is_test_file'] == False)
            & (data_df['is_blank'] == False)
            & (data_df['file-label'] == False)
            ]
    else:
        data_df = data_df[
            (data_df['is_test_file'] == False)
            & (data_df['is_blank'] == False)
            & (data_df['file-label'] == True)
            ]

    data_df = data_df.fillna('')

    export_df_to_files(
        data_df,
        file_src_dir,
        file_line_num_dir,
    )


def export_data_all_releases(
        dataset_name,
        project_name,
        all_train_releases,
        all_eval_releases,
):
    train_release = all_train_releases[
        project_name
    ]
    evaluation_releases = all_eval_releases[
        project_name
    ]

    export_ngram_data_each_release(
        dataset_name,
        train_release,
        True,
    )

    for release in evaluation_releases:
        export_ngram_data_each_release(
            dataset_name,
            release,
            False,
        )


def export_ngram_data_all_projects(
        dataset_name,
        all_train_releases,
        all_eval_releases,
        all_releases,
):
    for project_name in all_releases:
        export_data_all_releases(
            dataset_name,
            project_name,
            all_train_releases,
            all_eval_releases,
        )


def export_errorprone_data(
        dataset_name,
        project_name,
        all_eval_releases,
):
    base_original_data_dir = (
        f'../../../../dataset/{dataset_name}/'
        'processed/File-level/'
    )
    data_for_error_prone_dir = (
        f'../../../../dataset/{dataset_name}/'
        'ErrorProne_data/'
    )

    evaluation_releases = (
        all_eval_releases[project_name][0:]
    )

    for release in evaluation_releases:
        save_dir = (
                data_for_error_prone_dir
                + release
                + '/'
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        csv_path = (
                base_original_data_dir
                + release
                + '_ground-truth-files_dataset.csv'
        )

        data_df = read_ground_truth_file_level_csv(
            csv_path
        )
        data_df = data_df[
            data_df['Bug'] == True
            ]

        for filename, df in data_df.groupby('File'):
            if (
                    'test' in filename
                    or '.java' not in filename
            ):
                continue

            filename = filename.replace('/', '_')

            # code = list(df['SRC'])[0].strip()
            #
            # with open(
            #         save_dir + filename,
            #         'w',
            #         encoding='utf-8',
            # ) as file:
            #     file.write(code)

            code = str(df['SRC'].iloc[0])
            code = re.sub(r'\r+\n', '\n', code)
            code = code.replace('\r', '\n')

            with open(
                    save_dir + filename,
                    'w',
                    encoding='utf-8',
                    newline='\n',
            ) as file:
                file.write(code)


def export_error_prone_data_all_projects(
        dataset_name,
        all_eval_releases,
        all_releases,
):
    for project_name in all_releases:
        export_errorprone_data(
            dataset_name,
            project_name,
            all_eval_releases,
        )


def read_ground_truth_file_level_csv(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f'Processed file-level dataset not found: {csv_path}'
        )

    data_df = pd.read_csv(
        csv_path,
        encoding='utf-8',
    )

    required_columns = {
        'File',
        'Bug',
        'SRC',
    }
    missing_columns = (
            required_columns
            - set(data_df.columns)
    )
    if missing_columns:
        raise ValueError(
            'Missing required columns in processed file-level data: '
            f'{sorted(missing_columns)}'
        )

    data_df = data_df.copy()
    data_df['File'] = (
        data_df['File']
        .fillna('')
        .astype(str)
    )
    data_df['SRC'] = (
        data_df['SRC']
        .fillna('')
        .astype(str)
    )

    normalized_bug_labels = (
        data_df['Bug']
        .astype(str)
        .str.strip()
        .str.lower()
    )
    invalid_bug_labels = sorted(
        set(normalized_bug_labels)
        - {'true', 'false'}
    )
    if invalid_bug_labels:
        raise ValueError(
            'Invalid Bug labels in processed file-level data: '
            f'{invalid_bug_labels[:10]}'
        )

    data_df['Bug'] = (
            normalized_bug_labels == 'true'
    )

    print(
        'Loaded processed file-level dataset: '
        f'release_file={os.path.basename(csv_path)}, '
        f'records={len(data_df)}'
    )

    return data_df


for dataset_name, release_config in (
        datasets_release_config.items()
):
    all_train_releases = release_config[
        'all_train_releases'
    ]
    all_eval_releases = release_config[
        'all_eval_releases'
    ]
    all_releases = release_config[
        'all_releases'
    ]

    print(
        f'[{dataset_name}] Projects: '
        f'{list(all_releases.keys())}'
    )

     export_ngram_data_all_projects(
         dataset_name,
         all_train_releases,
         all_eval_releases,
         all_releases,
     )
    export_error_prone_data_all_projects(
        dataset_name,
        all_eval_releases,
        all_releases,
    )
