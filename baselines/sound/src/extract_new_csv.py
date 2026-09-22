import argparse
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from baselines.sound.src.utils.config import (
    GLANCE_PROJECT_RELEASE_LIST,
    LINEDP_PROJECT_RELEASE_LIST,
)
from tools import read_java_file_without_comments

warnings.simplefilter(
    action='ignore',
    category=pd.errors.PerformanceWarning,
)

DATASET_ROOT = '../../../dataset'
DATA_ROOT = '../Data'
RESULT_ROOT = '../Result'

FILE_LEVEL_PATH_SUFFIX = (
    '_ground-truth-files_dataset.csv'
)
LINE_LEVEL_PATH_SUFFIX = (
    '_defective_lines_dataset.csv'
)

DATASET_RELEASES = {
    'glance_dataset': GLANCE_PROJECT_RELEASE_LIST,
    'linedp_dataset': LINEDP_PROJECT_RELEASE_LIST,
}

DEFAULT_DATASET_NAME = 'linedp_dataset'
DEFAULT_CSV_CHUNK_SIZE = 256

FILE_TOKEN_ANALYZER = CountVectorizer(
    lowercase=False,
    min_df=1,
).build_analyzer()

LINE_TOKENIZER = CountVectorizer(
    lowercase=False,
    min_df=2,
).build_tokenizer()

WORD_PATTERN = re.compile(r'\b[a-zA-Z]{2,}\b')


@dataclass
class SparseTokenTable:

    rows: List[Dict[str, int]]
    columns: List[str]


def read_source_metadata(
        file_level_path: str,
) -> Tuple[List[str], List[int], List[str], List[int]]:

    with open(
            file_level_path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as input_file:
        lines = input_file.readlines()

    first_line_indices = {}
    for line_index, line in enumerate(lines):
        if line not in first_line_indices:
            first_line_indices[line] = line_index

    source_file_indices = [
        first_line_indices[line]
        for line in lines
        if (
                r'.java,true,"' in line
                or r'.java,false,"' in line
        )
    ]
    source_files = [
        lines[index].split(',')[0]
        for index in source_file_indices
    ]
    string_labels = [
        lines[index].split(',')[1]
        for index in source_file_indices
    ]
    numeric_labels = [
        1 if label == 'true' else 0
        for label in string_labels
    ]

    assert len(numeric_labels) == len(source_file_indices)

    return (
        lines,
        source_file_indices,
        source_files,
        numeric_labels,
    )


def dataframe_to_sparse_rows(
        dataframe: pd.DataFrame,
) -> Tuple[List[Dict[str, int]], List[str]]:

    columns = [
        str(column)
        for column in dataframe.columns
        if column != 'Bug'
    ]
    rows = []

    for row_values in dataframe.itertuples(
            index=False,
            name=None,
    ):
        row = {}
        for column, value in zip(
                dataframe.columns,
                row_values,
        ):
            if column == 'Bug':
                row['Bug'] = int(value)
            elif pd.notna(value) and float(value) != 0.0:
                row[str(column)] = int(value)

        if 'Bug' not in row:
            row['Bug'] = 0
        rows.append(row)

    return rows, columns


def resolve_column_order(
        rows: Sequence[Dict[str, int]],
        trailing_columns: Optional[Sequence[str]] = None,
) -> List[str]:

    columns = []
    seen_columns = set()

    for row in rows:
        for token in row:
            if token == 'Bug' or token in seen_columns:
                continue
            seen_columns.add(token)
            columns.append(token)

    if trailing_columns:
        for token in trailing_columns:
            if token == 'Bug' or token in seen_columns:
                continue
            seen_columns.add(token)
            columns.append(token)

    return columns


def build_sparse_table(
        rows: List[Dict[str, int]],
        columns: Optional[List[str]] = None,
) -> SparseTokenTable:

    if columns is None:
        columns = resolve_column_order(rows)

    return SparseTokenTable(
        rows=rows,
        columns=columns,
    )


def write_sparse_token_table(
        table: SparseTokenTable,
        output_path: str,
        chunk_size: int = DEFAULT_CSV_CHUNK_SIZE,
) -> None:

    if chunk_size <= 0:
        raise ValueError(
            'chunk_size must be greater than zero.'
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    row_count = len(table.rows)
    column_count = len(table.columns)
    column_index = {
        token: index
        for index, token in enumerate(table.columns)
    }

    sparse_row_indices = []
    sparse_column_indices = []
    sparse_values = []
    bug_labels = np.zeros(
        row_count,
        dtype=np.int64,
    )

    for row_index, row in enumerate(table.rows):
        bug_labels[row_index] = int(
            row.get('Bug', 0)
        )

        for token, value in row.items():
            if token == 'Bug' or int(value) == 0:
                continue

            sparse_row_indices.append(row_index)
            sparse_column_indices.append(
                column_index[token]
            )
            sparse_values.append(int(value))

    sparse_matrix = csr_matrix(
        (
            sparse_values,
            (
                sparse_row_indices,
                sparse_column_indices,
            ),
        ),
        shape=(row_count, column_count),
        dtype=np.int64,
    )

    if row_count == 0:
        empty_dataframe = pd.DataFrame(
            columns=table.columns + ['Bug']
        )
        empty_dataframe.to_csv(
            output_path,
            index=False,
        )
        return

    all_positive_mask = (
            np.asarray(
                sparse_matrix.getnnz(axis=0)
            ).ravel()
            == row_count
    )
    integer_columns = [
        table.columns[index]
        for index, is_all_positive
        in enumerate(all_positive_mask)
        if is_all_positive
    ]

    first_chunk = True

    for chunk_start in tqdm(
            range(0, row_count, chunk_size),
            desc=(
                    f'Writing {os.path.basename(output_path)}'
            ),
            unit='chunk',
            leave=False,
            dynamic_ncols=True,
    ):
        chunk_end = min(
            chunk_start + chunk_size,
            row_count,
        )

        dense_chunk = (
            sparse_matrix[
            chunk_start:chunk_end
            ]
            .toarray()
            .astype(
                np.float64,
                copy=False,
            )
        )
        chunk_dataframe = pd.DataFrame(
            dense_chunk,
            columns=table.columns,
        )

        if integer_columns:
            chunk_dataframe[
                integer_columns
            ] = chunk_dataframe[
                integer_columns
            ].astype('int64')

        chunk_dataframe['Bug'] = bug_labels[
                                 chunk_start:chunk_end
                                 ]

        chunk_dataframe.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk,
        )
        first_chunk = False


def combine_sparse_tables(
        first_table: SparseTokenTable,
        second_table: SparseTokenTable,
) -> SparseTokenTable:


    combined_columns = list(
        first_table.columns
    )
    seen_columns = set(combined_columns)

    for token in second_table.columns:
        if token not in seen_columns:
            seen_columns.add(token)
            combined_columns.append(token)

    return SparseTokenTable(
        rows=(
                list(first_table.rows)
                + list(second_table.rows)
        ),
        columns=combined_columns,
    )


def extract_token_row(
        lines: List[str],
        source_file_indices: List[int],
        numeric_labels: List[int],
        file_index: int,
        temporary_java_path: str,
) -> Optional[Dict[str, int]]:


    start_index = source_file_indices[file_index]
    end_index = (
        source_file_indices[file_index + 1]
        if file_index + 1 < len(source_file_indices)
        else len(lines)
    )

    code_lines = [
        line.strip()
        for line in lines[
                    start_index:end_index
                    ]
    ]

    if len(code_lines) == 0:
        return None

    code_lines[0] = (
        code_lines[0].split(',')[-1][1:]
    )
    code_lines[-1] = code_lines[-1][:-1]

    with open(
            temporary_java_path,
            'w',
            encoding='utf-8',
    ) as temporary_java_file:
        temporary_java_file.write(
            ''.join(code_lines)
        )

    java_content = read_java_file_without_comments(
        temporary_java_path
    )

    if java_content == '':
        return None

    token_counts = Counter(
        FILE_TOKEN_ANALYZER(java_content)
    )

    if not token_counts:

        raise ValueError(
            'empty vocabulary; perhaps the documents only '
            'contain stop words'
        )

    row = {
        token: int(token_counts[token])
        for token in sorted(token_counts)
        if token != 'Bug'
    }


    row['Bug'] = int(
        numeric_labels[file_index]
    )

    return row


def load_fallback_table(
        concat_file_path: str,
) -> Tuple[List[Dict[str, int]], List[str]]:


    fallback_dataframe = pd.read_csv(
        concat_file_path
    )
    return dataframe_to_sparse_rows(
        fallback_dataframe
    )


def build_token_part(
        release: str,
        lines: List[str],
        source_file_indices: List[int],
        numeric_labels: List[int],
        start_index: int,
        end_index: int,
        description: str,
        concat_file_path: str,
        temporary_java_path: str,
) -> SparseTokenTable:


    base_row = None
    pending_rows = []
    fallback_rows = None
    fallback_columns = None

    for file_index in tqdm(
            range(start_index, end_index),
            desc=f'[{release}] {description}',
            unit='file',
            leave=False,
            dynamic_ncols=True,
    ):
        row = extract_token_row(
            lines=lines,
            source_file_indices=source_file_indices,
            numeric_labels=numeric_labels,
            file_index=file_index,
            temporary_java_path=temporary_java_path,
        )

        if row is None:
            continue

        if file_index == start_index:
            base_row = row
        else:
            if (
                    base_row is None
                    and fallback_rows is None
            ):
                (
                    fallback_rows,
                    fallback_columns,
                ) = load_fallback_table(
                    concat_file_path
                )

            pending_rows.append(row)

    reversed_pending_rows = list(
        reversed(pending_rows)
    )

    if base_row is not None:
        final_rows = (
                reversed_pending_rows
                + [base_row]
        )
        final_columns = resolve_column_order(
            final_rows
        )
    elif fallback_rows is not None:
        final_rows = (
                reversed_pending_rows
                + fallback_rows
        )
        pending_columns = resolve_column_order(
            reversed_pending_rows
        )
        final_columns = resolve_column_order(
            reversed_pending_rows,
            trailing_columns=fallback_columns,
        )

        del pending_columns
    else:
        (
            final_rows,
            final_columns,
        ) = load_fallback_table(
            concat_file_path
        )

    return build_sparse_table(
        rows=final_rows,
        columns=final_columns,
    )


def extract_tokens_from(
        dataset_name: str,
        release: str,
        csv_chunk_size: int = DEFAULT_CSV_CHUNK_SIZE,
) -> None:


    dataset_input_root = os.path.join(
        DATASET_ROOT,
        dataset_name,
    )
    file_level_directory = os.path.join(
        dataset_input_root,
        'File-level',
    )
    dataset_data_root = os.path.join(
        DATA_ROOT,
        dataset_name,
    )
    tokens_directory = os.path.join(
        dataset_data_root,
        'tokens',
    )

    release_result_root = os.path.join(
        RESULT_ROOT,
        dataset_name,
        release,
    )

    os.makedirs(
        tokens_directory,
        exist_ok=True,
    )
    os.makedirs(
        release_result_root,
        exist_ok=True,
    )

    tokens_part1_path = os.path.join(
        tokens_directory,
        f'{release}_tokens_part1.csv',
    )
    tokens_part2_path = os.path.join(
        tokens_directory,
        f'{release}_tokens_part2.csv',
    )
    tokens_path = os.path.join(
        tokens_directory,
        f'{release}_tokens.csv',
    )
    concat_file_path = os.path.join(
        release_result_root,
        'concat.csv',
    )
    temporary_java_path = os.path.join(
        release_result_root,
        'tmp.java',
    )
    file_level_path = os.path.join(
        file_level_directory,
        f'{release}{FILE_LEVEL_PATH_SUFFIX}',
    )

    if not os.path.exists(file_level_path):
        raise FileNotFoundError(
            f'File-level dataset not found: '
            f'{file_level_path}'
        )

    (
        lines,
        source_file_indices,
        _,
        numeric_labels,
    ) = read_source_metadata(
        file_level_path
    )

    middle_index = int(
        len(source_file_indices) / 2
    )

    first_table = build_token_part(
        release=release,
        lines=lines,
        source_file_indices=source_file_indices,
        numeric_labels=numeric_labels,
        start_index=0,
        end_index=middle_index,
        description='Extracting tokens part 1/2',
        concat_file_path=concat_file_path,
        temporary_java_path=temporary_java_path,
    )
    write_sparse_token_table(
        first_table,
        concat_file_path,
        chunk_size=csv_chunk_size,
    )
    write_sparse_token_table(
        first_table,
        tokens_part1_path,
        chunk_size=csv_chunk_size,
    )

    second_table = build_token_part(
        release=release,
        lines=lines,
        source_file_indices=source_file_indices,
        numeric_labels=numeric_labels,
        start_index=middle_index,
        end_index=len(source_file_indices),
        description='Extracting tokens part 2/2',
        concat_file_path=concat_file_path,
        temporary_java_path=temporary_java_path,
    )
    write_sparse_token_table(
        second_table,
        concat_file_path,
        chunk_size=csv_chunk_size,
    )
    write_sparse_token_table(
        second_table,
        tokens_part2_path,
        chunk_size=csv_chunk_size,
    )

    combined_table = combine_sparse_tables(
        first_table,
        second_table,
    )
    write_sparse_token_table(
        combined_table,
        tokens_path,
        chunk_size=csv_chunk_size,
    )

    print(
        f'[{dataset_name}][{release}] '
        f'Saved token data to {tokens_path}'
    )


def get_tokens_split(
        tokens_file_path: str,
) -> set:

    token_columns = pd.read_csv(
        tokens_file_path,
        encoding='ISO-8859-1',
        nrows=0,
    ).columns
    return set(token_columns)


def get_n_score(
        dataset_name: str,
        release: str,
) -> None:

    dataset_input_root = os.path.join(
        DATASET_ROOT,
        dataset_name,
    )
    file_level_directory = os.path.join(
        dataset_input_root,
        'File-level',
    )
    line_level_directory = os.path.join(
        dataset_input_root,
        'Line-level',
    )
    dataset_data_root = os.path.join(
        DATA_ROOT,
        dataset_name,
    )
    tokens_directory = os.path.join(
        dataset_data_root,
        'tokens',
    )
    n_score_directory = os.path.join(
        dataset_data_root,
        'n_score',
    )

    os.makedirs(
        n_score_directory,
        exist_ok=True,
    )

    file_level_path = os.path.join(
        file_level_directory,
        f'{release}{FILE_LEVEL_PATH_SUFFIX}',
    )
    line_level_path = os.path.join(
        line_level_directory,
        f'{release}{LINE_LEVEL_PATH_SUFFIX}',
    )
    tokens_file_path = os.path.join(
        tokens_directory,
        f'{release}_tokens.csv',
    )
    cf_file_path = os.path.join(
        n_score_directory,
        f'{release}_cf.json',
    )
    cs_file_path = os.path.join(
        n_score_directory,
        f'{release}_cs.json',
    )

    if not os.path.exists(tokens_file_path):
        raise FileNotFoundError(
            f'Token file not found: '
            f'{tokens_file_path}'
        )
    if not os.path.exists(file_level_path):
        raise FileNotFoundError(
            f'File-level dataset not found: '
            f'{file_level_path}'
        )
    if not os.path.exists(line_level_path):
        raise FileNotFoundError(
            f'Line-level dataset not found: '
            f'{line_level_path}'
        )

    tokens = get_tokens_split(
        tokens_file_path
    )
    n_cf = {
        token: 0
        for token in tokens
    }
    n_cs = {
        token: 0
        for token in tokens
    }

    (
        lines,
        source_file_indices,
        source_files,
        _,
    ) = read_source_metadata(
        file_level_path
    )

    defective_lines_by_file = defaultdict(
        list
    )
    with open(
            line_level_path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as line_level_file:
        line_level_lines = (
            line_level_file.readlines()
        )

    for line in line_level_lines[1:]:
        values = line.split(',', 2)
        bug_line_file_name = values[0]
        bug_line = values[2]
        defective_lines_by_file[
            bug_line_file_name
        ].append(bug_line)

    token_cache = {}
    tokens_contains = tokens.__contains__
    tokenizer = LINE_TOKENIZER
    word_search = WORD_PATTERN.search

    def get_matching_tokens(text: str):
        matching_tokens = token_cache.get(text)
        if matching_tokens is None:
            matching_tokens = {
                token
                for token in tokenizer(text)
                if tokens_contains(token)
            }
            token_cache[text] = matching_tokens
        return matching_tokens

    for file_index in tqdm(
            range(len(source_file_indices)),
            desc=f'[{release}] Calculating N-score',
            unit='file',
            leave=False,
            dynamic_ncols=True,
    ):
        file_name = source_files[file_index]
        start_index = source_file_indices[
            file_index
        ]
        end_index = (
            source_file_indices[file_index + 1]
            if (
                    file_index + 1
                    < len(source_file_indices)
            )
            else len(lines)
        )

        file_lines = [
            line.strip()
            for line in lines[
                        start_index:end_index
                        ]
        ]
        file_lines[0] = (
            file_lines[0].split(',')[-1][1:]
        )
        file_lines[-1] = file_lines[-1][:-1]

        file_lines = [
            line + '\n'
            for line in file_lines
        ]
        buggy_line_counts = Counter(
            defective_lines_by_file.get(
                file_name,
                [],
            )
        )

        for line in file_lines:
            stripped_line = line.strip()

            if buggy_line_counts[line] > 0:
                text_for_regex = stripped_line
                text_for_tokenizer = line
                is_buggy = True
                buggy_line_counts[line] -= 1
            elif (
                    buggy_line_counts[
                        stripped_line
                    ] > 0
            ):
                text_for_regex = stripped_line
                text_for_tokenizer = line
                is_buggy = True
                buggy_line_counts[
                    stripped_line
                ] -= 1
            else:
                csv_quoted_line = (
                        '"'
                        + stripped_line.replace(
                    '"',
                    '""',
                )
                        + '"\n'
                )

                if (
                        buggy_line_counts[
                            csv_quoted_line
                        ] > 0
                ):
                    text_for_regex = (
                        csv_quoted_line.strip()
                    )
                    text_for_tokenizer = (
                        csv_quoted_line
                    )
                    is_buggy = True
                    buggy_line_counts[
                        csv_quoted_line
                    ] -= 1
                else:
                    text_for_regex = (
                        stripped_line
                    )
                    text_for_tokenizer = line
                    is_buggy = False

            if word_search(text_for_regex):
                target = n_cf if is_buggy else n_cs

                for token in get_matching_tokens(
                        text_for_tokenizer
                ):
                    target[token] += 1

    with open(
            cf_file_path,
            'w',
            encoding='utf-8',
    ) as cf_file:
        json.dump(n_cf, cf_file)

    with open(
            cs_file_path,
            'w',
            encoding='utf-8',
    ) as cs_file:
        json.dump(n_cs, cs_file)

    print(
        f'[{dataset_name}][{release}] '
        f'Saved N-score counts to {n_score_directory}'
    )


def get_release_output_paths(
        dataset_name: str,
        release: str,
) -> Tuple[str, str, str, str, str]:

    dataset_data_root = os.path.join(
        DATA_ROOT,
        dataset_name,
    )
    token_path = os.path.join(
        dataset_data_root,
        'tokens',
        f'{release}_tokens.csv',
    )
    cf_path = os.path.join(
        dataset_data_root,
        'n_score',
        f'{release}_cf.json',
    )
    cs_path = os.path.join(
        dataset_data_root,
        'n_score',
        f'{release}_cs.json',
    )
    token_complete_path = (
            token_path + '.complete'
    )
    n_score_complete_path = os.path.join(
        dataset_data_root,
        'n_score',
        f'{release}_n_score.complete',
    )

    return (
        token_path,
        cf_path,
        cs_path,
        token_complete_path,
        n_score_complete_path,
    )


def write_completion_marker(
        marker_path: str,
) -> None:

    os.makedirs(
        os.path.dirname(marker_path),
        exist_ok=True,
    )
    with open(
            marker_path,
            'w',
            encoding='utf-8',
    ) as marker_file:
        marker_file.write('complete\n')


def process_release(
        dataset_name: str,
        release: str,
        csv_chunk_size: int,
        skip_existing: bool,
) -> str:

    (
        token_path,
        cf_path,
        cs_path,
        token_complete_path,
        n_score_complete_path,
    ) = get_release_output_paths(
        dataset_name,
        release,
    )

    token_stage_is_complete = (
            os.path.exists(token_path)
            and os.path.exists(
        token_complete_path
    )
    )

    if not (
            skip_existing
            and token_stage_is_complete
    ):
        extract_tokens_from(
            dataset_name=dataset_name,
            release=release,
            csv_chunk_size=csv_chunk_size,
        )
        write_completion_marker(
            token_complete_path
        )
    else:
        print(
            f'[{dataset_name}][{release}] '
            'Skipping completed token data.'
        )

    n_score_stage_is_complete = (
            os.path.exists(cf_path)
            and os.path.exists(cs_path)
            and os.path.exists(
        n_score_complete_path
    )
    )

    if not (
            skip_existing
            and n_score_stage_is_complete
    ):
        get_n_score(
            dataset_name=dataset_name,
            release=release,
        )
        write_completion_marker(
            n_score_complete_path
        )
    else:
        print(
            f'[{dataset_name}][{release}] '
            'Skipping completed N-score data.'
        )

    return release


def process_release_task(
        task: Tuple[str, str, int, bool],
) -> str:

    return process_release(*task)


def process_dataset(
        dataset_name: str,
        workers: int,
        csv_chunk_size: int,
        skip_existing: bool,
) -> None:

    if dataset_name not in DATASET_RELEASES:
        raise ValueError(
            f'Unsupported dataset: {dataset_name}'
        )

    releases = list(
        DATASET_RELEASES[dataset_name]
    )

    if workers == 1:
        for release in tqdm(
                releases,
                desc=(
                        f'Processing {dataset_name} releases'
                ),
                unit='release',
        ):
            process_release(
                dataset_name=dataset_name,
                release=release,
                csv_chunk_size=csv_chunk_size,
                skip_existing=skip_existing,
            )
        return

    tasks = [
        (
            dataset_name,
            release,
            csv_chunk_size,
            skip_existing,
        )
        for release in releases
    ]

    with ProcessPoolExecutor(
            max_workers=workers
    ) as executor:
        future_to_release = {
            executor.submit(
                process_release_task,
                task,
            ): task[1]
            for task in tasks
        }

        with tqdm(
                total=len(tasks),
                desc=(
                        f'Processing {dataset_name} releases'
                ),
                unit='release',
        ) as progress_bar:
            for future in as_completed(
                    future_to_release
            ):
                release = future_to_release[
                    future
                ]
                future.result()
                progress_bar.update(1)
                progress_bar.set_postfix_str(
                    release
                )


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            'Extract source-code tokens and calculate N-scores '
            'for GLANCE_Dataset or LINEDP_Dataset.'
        )
    )
    parser.add_argument(
        '--dataset',
        choices=(
            'glance_dataset',
            'linedp_dataset',
            'all',
        ),
        default=DEFAULT_DATASET_NAME,
        help=(
            'Dataset to process. The default is linedp_dataset.'
        ),
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help=(
            'Number of releases processed in parallel. '
            'Use 1 for minimum memory usage.'
        ),
    )
    parser.add_argument(
        '--csv-chunk-size',
        type=int,
        default=DEFAULT_CSV_CHUNK_SIZE,
        help=(
            'Number of dense rows written per CSV chunk.'
        ),
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help=(
            'Skip stages carrying a completion marker from a '
            'previous successful optimized run.'
        ),
    )

    arguments = parser.parse_args()

    if arguments.workers <= 0:
        parser.error(
            '--workers must be greater than zero.'
        )
    if arguments.csv_chunk_size <= 0:
        parser.error(
            '--csv-chunk-size must be greater than zero.'
        )

    return arguments


def main():

    arguments = parse_arguments()

    if arguments.dataset == 'all':
        selected_datasets = tuple(
            DATASET_RELEASES.keys()
        )
    else:
        selected_datasets = (
            arguments.dataset,
        )

    for dataset_name in selected_datasets:
        process_dataset(
            dataset_name=dataset_name,
            workers=arguments.workers,
            csv_chunk_size=(
                arguments.csv_chunk_size
            ),
            skip_existing=(
                arguments.skip_existing
            ),
        )


if __name__ == '__main__':
    main()
