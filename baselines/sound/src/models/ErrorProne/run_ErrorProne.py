import os
import re
import subprocess

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

glance_eval_releases = [
    'ambari-2.1.0', 'ambari-2.2.0', 'ambari-2.4.0',
    'ambari-2.5.0', 'ambari-2.6.0', 'ambari-2.7.0',
    'amq-5.1.0', 'amq-5.2.0', 'amq-5.4.0',
    'amq-5.5.0', 'amq-5.6.0', 'amq-5.7.0',
    'amq-5.8.0', 'amq-5.9.0', 'amq-5.10.0',
    'amq-5.11.0', 'amq-5.12.0', 'amq-5.14.0',
    'amq-5.15.0',
    'bookkeeper-4.2.0', 'bookkeeper-4.4.0',
    'calcite-1.8.0', 'calcite-1.11.0', 'calcite-1.13.0',
    'calcite-1.15.0', 'calcite-1.16.0', 'calcite-1.17.0',
    'calcite-1.18.0',
    'cassandra-0.8.6', 'cassandra-1.0.9',
    'cassandra-1.1.6', 'cassandra-1.1.11',
    'cassandra-1.2.11',
    'flink-1.6.0',
    'groovy-1.5.5', 'groovy-1.6.0', 'groovy-1.7.3',
    'groovy-1.7.6', 'groovy-1.8.1', 'groovy-1.8.7',
    'groovy-2.1.0', 'groovy-2.1.6', 'groovy-2.4.4',
    'groovy-2.4.6', 'groovy-2.4.8', 'groovy-2.5.0',
    'groovy-2.5.5',
    'hbase-0.94.5', 'hbase-0.98.0', 'hbase-0.98.5',
    'hbase-0.98.11',
    'hive-1.2.0', 'hive-2.0.0', 'hive-2.1.0',
    'ignite-1.4.0', 'ignite-1.6.0',
    'log4j2-2.1', 'log4j2-2.2', 'log4j2-2.3',
    'log4j2-2.4', 'log4j2-2.5', 'log4j2-2.6',
    'log4j2-2.7', 'log4j2-2.8', 'log4j2-2.9',
    'log4j2-2.10',
    'mahout-0.4', 'mahout-0.5', 'mahout-0.6',
    'mahout-0.7', 'mahout-0.8',
    'mng-3.2.0', 'mng-3.3.0', 'mng-3.5.0',
    'mng-3.6.0',
    'nifi-1.2.0', 'nifi-1.5.0', 'nifi-1.8.0',
    'nutch-1.3', 'nutch-1.4', 'nutch-1.5',
    'nutch-1.6', 'nutch-1.7', 'nutch-1.8',
    'nutch-1.9', 'nutch-1.10', 'nutch-1.12',
    'nutch-1.13', 'nutch-1.14', 'nutch-1.15',
    'storm-0.9.3', 'storm-1.0.0', 'storm-1.0.3',
    'storm-1.0.5',
    'tika-0.8', 'tika-0.9', 'tika-0.10',
    'tika-1.1', 'tika-1.3', 'tika-1.5',
    'tika-1.7', 'tika-1.10', 'tika-1.13',
    'tika-1.15', 'tika-1.17',
    'ww-2.0.5', 'ww-2.0.10', 'ww-2.1.1',
    'ww-2.1.3', 'ww-2.1.7', 'ww-2.2.0',
    'ww-2.2.2', 'ww-2.3.1', 'ww-2.3.4',
    'ww-2.3.10', 'ww-2.3.15', 'ww-2.3.17',
    'ww-2.3.20', 'ww-2.3.24',
    'zookeeper-3.5.1', 'zookeeper-3.5.2',
    'zookeeper-3.5.3',
]

linedp_eval_releases = [
    'activemq-5.1.0', 'activemq-5.2.0',
    'activemq-5.3.0', 'activemq-5.8.0',
    'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0',
    'derby-10.3.1.4', 'derby-10.5.1.1',
    'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2',
    'hbase-0.95.0', 'hbase-0.95.2',
    'hive-0.10.0', 'hive-0.12.0',
    'jruby-1.4.0', 'jruby-1.5.0',
    'jruby-1.7.0.preview1',
    'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1',
    'wicket-1.3.0-beta2', 'wicket-1.5.3',
]

DATASET_EVAL_RELEASES = {
    # 'glance_dataset': glance_eval_releases,
    'linedp_dataset': linedp_eval_releases,
}

DATASET_ROOT = '../../../../../dataset'
RESULT_ROOT = './ErrorProne_result'

BASE_COMMAND = (
    'javac '
    '-J-Duser.language=en '
    '-J-Duser.country=US '
    '-J-Xbootclasspath/p:javac-9+181-r4173-1.jar '
    '-XDcompilePolicy=simple '
    '-processorpath '
    'error_prone_core-2.4.0-with-dependencies.jar:'
    'dataflow-shaded-3.1.2.jar:'
    'jFormatString-3.0.0.jar '
)


def run_ErrorProne(task):
    dataset_name, release = task

    base_file_dir = os.path.join(
        DATASET_ROOT,
        dataset_name,
        'ErrorProne_data',
    )
    result_dir = os.path.join(
        RESULT_ROOT,
        dataset_name,
    )
    os.makedirs(result_dir, exist_ok=True)

    dataframe_list = []
    java_file_dir = os.path.join(
        base_file_dir,
        release,
    )

    file_list = os.listdir(java_file_dir)

    for java_filename in tqdm(
            file_list,
            desc=f'{dataset_name} - {release}',
            unit='file',
            dynamic_ncols=True,
            leave=False,
    ):
        java_file_path = os.path.join(
            java_file_dir,
            java_filename,
        )

        with open(
                java_file_path,
                'r',
                encoding='utf-8',
                errors='ignore',
        ) as java_file:
            java_code = java_file.readlines()

        code_length = len(java_code)

        output = subprocess.getoutput(
            BASE_COMMAND + java_file_path
        )

        reported_lines = re.findall(
            r'\d+: error:',
            output,
        )
        reported_lines = [
            int(
                line.replace(':', '')
                .replace('error', '')
            )
            for line in reported_lines
        ]
        reported_lines = list(set(reported_lines))

        line_dataframe = pd.DataFrame()
        line_dataframe['filename'] = [
                                         java_filename.replace('_', '/')
                                     ] * code_length
        line_dataframe['test-release'] = [
                                             release
                                         ] * code_length
        line_dataframe['line_number'] = np.arange(
            1,
            code_length + 1,
        )
        line_dataframe['EP_prediction_result'] = (
            line_dataframe['line_number'].isin(
                reported_lines
            )
        )

        dataframe_list.append(line_dataframe)

    if dataframe_list:
        final_dataframe = pd.concat(
            dataframe_list
        )
        output_path = os.path.join(
            result_dir,
            release + '-line-lvl-result.txt',
        )
        final_dataframe.to_csv(
            output_path,
            index=False,
        )

    print(
        f'[{dataset_name}] Finished {release}'
    )


if __name__ == '__main__':
    agents = 5
    chunksize = 8

    tasks = [
        (dataset_name, release)
        for dataset_name, releases
        in DATASET_EVAL_RELEASES.items()
        for release in releases
    ]

    with Pool(processes=agents) as pool:
        pool.map(
            run_ErrorProne,
            tasks,
            chunksize,
        )
