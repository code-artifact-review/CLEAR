import os
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import json
import warnings

from baselines.sound.src.tools import read_java_file_without_comments
from clear.src.my_utils.helper import glance_projects, linedp_projects

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

dataset_string = '../../dataset'
result_string = '.././Result'

os.makedirs(result_string, exist_ok=True)

file_bow_vectorizer = CountVectorizer(lowercase=False)
code_tokenizer = file_bow_vectorizer.build_tokenizer()

file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


def extract_file_level_bag_of_tokens(dataset_name, release):

    file_level_path = f'{dataset_string}/{dataset_name}/File-level/'

    bag_of_tokens_file_path = (
        f'../Data/{dataset_name}/tokens/{release}_tokens.csv'
    )
    os.makedirs(os.path.dirname(bag_of_tokens_file_path), exist_ok=True)

    file_level_dataset_path = (
        f'{file_level_path}{release}{file_level_path_suffix}'
    )
    with open(
            file_level_dataset_path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as file:
        lines = file.readlines()

    source_file_start_indices = [
        lines.index(line)
        for line in lines
        if (
                r'.java,true,"' in line
                or r'.java,false,"' in line
                or r'.java,True,"' in line
                or r'.java,False,"' in line
        )
    ]
    source_files = [
        lines[index].split(',')[0]
        for index in source_file_start_indices
    ]
    file_label_strings = [
        lines[index].split(',')[1]
        for index in source_file_start_indices
    ]
    file_labels = [
        1 if label.lower() == 'true' else 0
        for label in file_label_strings
    ]

    token_rows = []

    for file_index in tqdm(
            range(len(source_file_start_indices)),
            desc=f'{release} - Extracting BoT',
            position=1,
            leave=False,
            dynamic_ncols=True,
    ):
        start_index = source_file_start_indices[file_index]
        end_index = (
            source_file_start_indices[file_index + 1]
            if file_index + 1 < len(source_file_start_indices)
            else len(lines)
        )

        code_lines = [
            line.strip()
            for line in lines[start_index:end_index]
        ]
        if len(code_lines) == 0:
            continue

        code_lines[0] = (
            code_lines[0].split(',')[-1][1:]
            if ',' in code_lines[0]
            else code_lines[0]
        )
        code_lines[-1] = (
            code_lines[-1][:-1]
            if len(code_lines[-1]) > 0
            else code_lines[-1]
        )

        temporary_java_file_path = (
            f'{result_string}/{dataset_name}/tmp.java'
        )
        os.makedirs(
            os.path.dirname(temporary_java_file_path),
            exist_ok=True,
        )

        with open(
                temporary_java_file_path,
                'w',
                encoding='utf-8',
        ) as temporary_java_file:
            for code_line in code_lines:
                print(code_line, end='', file=temporary_java_file)

        java_content = read_java_file_without_comments(
            temporary_java_file_path
        )
        if not java_content or java_content.strip() == '':
            continue

        try:
            token_matrix = file_bow_vectorizer.fit_transform(
                [java_content]
            )
        except ValueError as error:
            print(
                f'Skipped file {source_files[file_index]}: {error}'
            )
            continue

        token_names = file_bow_vectorizer.get_feature_names_out()
        token_counts = token_matrix.toarray()[0]

        for token, token_count in zip(token_names, token_counts):
            token_rows.append({
                'filename': source_files[file_index],
                'token': token,
                'count': token_count,
                'Bug': file_labels[file_index],
            })

    bag_of_tokens_dataframe = pd.DataFrame(token_rows)
    bag_of_tokens_dataframe.to_csv(
        bag_of_tokens_file_path,
        sep=',',
        index=False,
        header=True,
    )


def load_token_vocabulary(bag_of_tokens_file_path) -> set:


    bag_of_tokens_dataframe = pd.read_csv(
        bag_of_tokens_file_path,
        encoding='utf-8',
    )
    return set(bag_of_tokens_dataframe['token'])


def collect_token_spectrum_information(dataset_name, release):


    file_level_directory = (
        f'{dataset_string}/{dataset_name}/File-level'
    )
    line_level_directory = (
        f'{dataset_string}/{dataset_name}/Line-level'
    )

    file_level_dataset_name = (
            release + '_ground-truth-files_dataset.csv'
    )
    line_level_dataset_name = (
            release + '_defective_lines_dataset.csv'
    )

    bag_of_tokens_file_path = (
        f'../Data/{dataset_name}/tokens/{release}_tokens.csv'
    )

    defective_spectrum_file_path = (
        f'../Data/{dataset_name}/n_score/{release}_cf.json'
    )
    clean_spectrum_file_path = (
        f'../Data/{dataset_name}/n_score/{release}_cs.json'
    )

    os.makedirs(
        os.path.dirname(clean_spectrum_file_path),
        exist_ok=True,
    )

    token_vocabulary = load_token_vocabulary(
        bag_of_tokens_file_path
    )

    defective_line_count_by_token = {
        token: 0
        for token in token_vocabulary
    }
    clean_line_count_by_token = {
        token: 0
        for token in token_vocabulary
    }

    file_level_dataset_path = (
        f'{file_level_directory}/{file_level_dataset_name}'
    )
    with open(
            file_level_dataset_path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as file:
        lines = file.readlines()

    source_file_start_indices = [
        lines.index(line)
        for line in lines
        if (
                r'.java,true,"' in line
                or r'.java,false,"' in line
                or r'.java,True,"' in line
                or r'.java,False,"' in line
        )
    ]
    source_files = [
        lines[index].split(',')[0]
        for index in source_file_start_indices
    ]

    defective_lines_by_file = {}

    line_level_dataset_path = (
        f'{line_level_directory}/{line_level_dataset_name}'
    )
    with open(
            line_level_dataset_path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as file:
        line_level_lines = file.readlines()

        for line in line_level_lines[1:]:
            columns = line.split(',', 2)
            if len(columns) < 3:
                continue

            defective_file_name, _, defective_code_line = columns
            defective_code_line = defective_code_line.rstrip('\n')

            if defective_file_name not in defective_lines_by_file:
                defective_lines_by_file[defective_file_name] = set()

            defective_lines_by_file[defective_file_name].add(
                defective_code_line
            )

    for file_index in tqdm(
            range(len(source_file_start_indices)),
            desc=f'{release} - Collecting token spectrum',
            position=2,
            leave=False,
            dynamic_ncols=True,
    ):
        file_name = source_files[file_index]
        start_index = source_file_start_indices[file_index]
        end_index = (
            source_file_start_indices[file_index + 1]
            if file_index + 1 < len(source_file_start_indices)
            else len(lines)
        )

        file_lines = [
            line.strip()
            for line in lines[start_index:end_index]
        ]
        if not file_lines:
            continue

        file_lines[0] = file_lines[0].split(',')[-1][1:]
        file_lines[-1] = file_lines[-1][:-1]

        defective_lines = defective_lines_by_file.get(
            file_name,
            set(),
        )

        for raw_code_line in file_lines:
            code_line = raw_code_line.strip()
            escaped_code_line = code_line.replace('"', '""')

            is_defective_line = (
                    raw_code_line in defective_lines
                    or code_line in defective_lines
                    or f'"{escaped_code_line}"\n' in defective_lines
            )

            if re.findall(r'\b[a-zA-Z]{2,}\b', code_line):
                line_token_set = set(code_tokenizer(code_line))
                matched_tokens = (
                        line_token_set & token_vocabulary
                )

                for token in matched_tokens:
                    if is_defective_line:
                        defective_line_count_by_token[token] += 1
                    else:
                        clean_line_count_by_token[token] += 1

    with open(
            defective_spectrum_file_path,
            'w',
    ) as file:
        json.dump(defective_line_count_by_token, file)

    with open(
            clean_spectrum_file_path,
            'w',
    ) as file:
        json.dump(clean_line_count_by_token, file)


if __name__ == '__main__':
    datasets_projects = {
        # 'glance_dataset': glance_projects,
        'linedp_dataset': linedp_projects,
    }

    for dataset_name, projects in datasets_projects.items():
        for project, releases in projects.items():
            for release in releases:
                extract_file_level_bag_of_tokens(
                    dataset_name,
                    release,
                )
                collect_token_spectrum_information(
                    dataset_name,
                    release,
                )
