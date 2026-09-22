import shutil
from warnings import simplefilter
from pathlib import Path
import os
import re
import numpy as np
import pickle

simplefilter(action='ignore', category=FutureWarning)

linedp_projects = {
    'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
    'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
    'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
    'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
    'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
    'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
    'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
    'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
    'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']}

glance_projects = {

    'ambari': ['ambari-1.2.0', 'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0', 'ambari-2.5.0', 'ambari-2.6.0',
               'ambari-2.7.0'],
    'amq': ['amq-5.0.0', 'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0', 'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0', 'amq-5.8.0',
            'amq-5.9.0', 'amq-5.10.0', 'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0', 'amq-5.15.0'],
    'bookkeeper': ['bookkeeper-4.0.0', 'bookkeeper-4.2.0', 'bookkeeper-4.4.0'],
    'calcite': ['calcite-1.6.0', 'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0', 'calcite-1.15.0',
                'calcite-1.16.0', 'calcite-1.17.0', 'calcite-1.18.0'],
    'cassandra': ['cassandra-0.7.4', 'cassandra-0.8.6', 'cassandra-1.0.9', 'cassandra-1.1.6', 'cassandra-1.1.11',
                  'cassandra-1.2.11'],
    'flink': ['flink-1.4.0', 'flink-1.6.0'],
    'groovy': ['groovy-1.0', 'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3', 'groovy-1.7.6', 'groovy-1.8.1',
               'groovy-1.8.7', 'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4', 'groovy-2.4.6', 'groovy-2.4.8',
               'groovy-2.5.0', 'groovy-2.5.5'],
    'hbase': ['hbase-0.94.1', 'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5', 'hbase-0.98.11'],
    'hive': ['hive-0.14.0', 'hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0'],
    'ignite': ['ignite-1.0.0', 'ignite-1.4.0', 'ignite-1.6.0'],
    'log4j2': ['log4j2-2.0', 'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3', 'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6',
               'log4j2-2.7', 'log4j2-2.8', 'log4j2-2.9', 'log4j2-2.10'],
    'mahout': ['mahout-0.3', 'mahout-0.4', 'mahout-0.5', 'mahout-0.6', 'mahout-0.7', 'mahout-0.8'],
    'mng': ['mng-3.0.0', 'mng-3.1.0', 'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0', 'mng-3.6.0'],
    'nifi': ['nifi-0.4.0', 'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0'],
    'nutch': ['nutch-1.1', 'nutch-1.3', 'nutch-1.4', 'nutch-1.5', 'nutch-1.6', 'nutch-1.7', 'nutch-1.8', 'nutch-1.9',
              'nutch-1.10', 'nutch-1.12', 'nutch-1.13', 'nutch-1.14', 'nutch-1.15'],
    'storm': ['storm-0.9.0', 'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3', 'storm-1.0.5'],
    'tika': ['tika-0.7', 'tika-0.8', 'tika-0.9', 'tika-0.10', 'tika-1.1', 'tika-1.3', 'tika-1.5', 'tika-1.7',
             'tika-1.10', 'tika-1.13', 'tika-1.15', 'tika-1.17'],
    'ww': ['ww-2.0.0', 'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1', 'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0', 'ww-2.2.2', 'ww-2.3.1',
           'ww-2.3.4', 'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17', 'ww-2.3.20', 'ww-2.3.24'],
    'zookeeper': ['zookeeper-3.4.6', 'zookeeper-3.5.1', 'zookeeper-3.5.2', 'zookeeper-3.5.3']
}

root_path = str(Path(__file__).resolve().parents[3]) + '/'
dataset_string = 'dataset'
result_string = 'Result'

dataset_path = f'{root_path}/{dataset_string}/Bug-Info/'

result_path = f'{root_path}{result_string}'
file_level_path_suffix = '_ground-truth-files_dataset.csv'
line_level_path_suffix = '_defective_lines_dataset.csv'


def read_file_level_dataset(which_dataset, release):
    file_path = f'{root_path}{dataset_string}/{which_dataset}/File-level/'
    path = f'{file_path}{release}{file_level_path_suffix}'
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        src_file_indices = [lines.index(line) for line in lines if
                            r'.java,true,"' in line or r'.java,false,"' in line or r'.java,True,"' in line or r'.java,False,"' in line]
        src_files = [lines[index].split(',')[0] for index in src_file_indices]
        string_labels = [lines[index].split(',')[1] for index in src_file_indices]
        numeric_labels = [1 if label.lower() == 'true' else 0 for label in string_labels]

        texts_lines = []
        texts_lines_without_comments = []
        for i in range(len(src_file_indices)):
            s_index = src_file_indices[i]
            e_index = src_file_indices[i + 1] if i + 1 < len(src_file_indices) else len(lines)

            code_lines = [line.strip() for line in lines[s_index:e_index]]
            code_lines[0] = code_lines[0].split(',')[-1][1:]
            code_lines = code_lines[:-1]
            texts_lines.append(code_lines)

            enumerated_lines = list(enumerate(code_lines))
            new_lines = []
            for index, line in enumerated_lines:
                if not (line.startswith('/') or line.startswith('*')):
                    new_lines.append(line)

            code_lines_without_comments = [line.strip() for line in new_lines]
            texts_lines_without_comments.append(code_lines_without_comments)

        texts = [' '.join(line) for line in texts_lines]

        return texts, texts_lines, numeric_labels, src_files, texts_lines_without_comments


def read_line_level_dataset(which_dataset, release):
    line_level_path = f'{root_path}{dataset_string}/{which_dataset}/Line-level/'
    path = f'{line_level_path}{release}{line_level_path_suffix}'
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
        file_buggy_lines_dict = {}
        for line in lines[1:]:
            temp = line.split(',', 2)
            file_name, buggy_line_number = temp[0], int(temp[1])
            if file_name not in file_buggy_lines_dict.keys():
                file_buggy_lines_dict[file_name] = [buggy_line_number]
            else:
                file_buggy_lines_dict[file_name].append(buggy_line_number)

    return file_buggy_lines_dict


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_csv_result(file_path, file_name, data):
    make_path(file_path)
    with open(f'{file_path}{file_name}', 'w', encoding='utf-8') as file:
        file.write(data)
    print(f'Result has been saved to {file_path}{file_name} successfully!')
