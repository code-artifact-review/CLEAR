import math
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOUND_ROOT = Path(__file__).resolve().parents[2]

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.metrics import roc_auc_score

# from sound.src.utils.helper import *
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from baselines.sound.src.utils.config import (
    GLANCE_PROJECT_RELEASE_LIST,
    LINEDP_PROJECT_RELEASE_LIST,
)
from baselines.sound.src.utils.helper import *

USE_CACHE = False


def calc_auc(label, pred):
    if hasattr(label, "detach"):
        label = label.detach().cpu().numpy()

    if hasattr(pred, "detach"):
        pred = pred.detach().cpu().numpy()

    label = np.asarray(label).reshape(-1)
    pred = np.asarray(pred)

    if pred.ndim == 2:
        if pred.shape[1] != 2:
            raise ValueError(
                f"Expected prediction shape (n_samples, 2), but got {pred.shape}."
            )
        pred = pred[:, 1]
    else:
        pred = pred.reshape(-1)

    if len(label) != len(pred):
        raise ValueError(
            f"Label and prediction lengths differ: {len(label)} != {len(pred)}."
        )

    if len(label) == 0:
        raise ValueError("Label and prediction arrays must not be empty.")

    if not np.all(np.isfinite(pred)):
        raise ValueError("Prediction scores contain NaN or infinite values.")

    unique_labels = np.unique(label)
    if len(unique_labels) != 2:
        raise ValueError(
            f"ROC-AUC requires exactly two classes, but found {unique_labels.tolist()}."
        )

    return float(roc_auc_score(label, pred))


def resolve_dataset_name(
        train_release: str,
        test_release: str,
) -> str:
    glance_releases = set(
        GLANCE_PROJECT_RELEASE_LIST
    )
    linedp_releases = set(
        LINEDP_PROJECT_RELEASE_LIST
    )

    if (
            train_release in glance_releases
            and test_release in glance_releases
    ):
        return 'glance_dataset'

    if (
            train_release in linedp_releases
            and test_release in linedp_releases
    ):
        return 'linedp_dataset'

    raise ValueError(
        'Cannot resolve one dataset for the release pair: '
        f'{train_release} -> {test_release}.'
    )


def resolve_dataset_root_path(
        dataset_name: str,
) -> str:
    candidate_roots = [
        PROJECT_ROOT / 'dataset',
        PROJECT_ROOT / 'Dataset',
    ]

    checked_locations = []

    for candidate_root in candidate_roots:
        dataset_root = candidate_root / dataset_name
        checked_locations.append(str(dataset_root))

        file_level_directory = dataset_root / 'File-level'
        line_level_directory = dataset_root / 'Line-level'

        if (
                file_level_directory.is_dir()
                and line_level_directory.is_dir()
        ):
            return str(candidate_root)

    raise FileNotFoundError(
        'Cannot locate the dataset-specific File-level and '
        'Line-level directories. Checked: '
        + '; '.join(checked_locations)
    )


def read_line_level_dataset_from_path(
        release: str,
        file_path: str,
) -> dict:
    if release == '':
        return {}

    path = os.path.join(
        file_path,
        release + line_level_path_suffix,
    )

    with open(
            path,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as input_file:
        lines = input_file.readlines()

    file_buggy_lines_dict = {}

    for line in lines[1:]:
        values = line.split(',', 2)
        file_name = values[0]
        buggy_line_number = int(values[1])

        if file_name not in file_buggy_lines_dict:
            file_buggy_lines_dict[file_name] = [
                buggy_line_number
            ]
        else:
            file_buggy_lines_dict[file_name].append(
                buggy_line_number
            )

    return file_buggy_lines_dict


class BaseModel(object):
    model_name = 'BaseModel'

    def __init__(self, train_release: str = '', test_release: str = '', test_result_path='', is_realistic=False):
        self.dataset_name = resolve_dataset_name(
            train_release,
            test_release,
        )

        if test_result_path != '':
            self.result_path = test_result_path
        else:
            self.result_path = os.path.join(
                str(PROJECT_ROOT / 'Result'),
                self.dataset_name,
                self.model_name,
            ) + os.sep

        self.file_level_result_path = os.path.join(
            self.result_path,
            'file_result',
        ) + os.sep
        self.line_level_result_path = os.path.join(
            self.result_path,
            'line_result',
        ) + os.sep
        self.buggy_density_path = os.path.join(
            self.result_path,
            'buggy_density',
        ) + os.sep

        self.file_level_evaluation_file = os.path.join(
            self.file_level_result_path,
            'evaluation.csv',
        )
        self.line_level_evaluation_file = os.path.join(
            self.line_level_result_path,
            'evaluation.csv',
        )
        self.execution_time_file = os.path.join(
            self.result_path,
            'time.csv',
        )

        self.dataset_data_path = str(
            SOUND_ROOT / 'Data' / self.dataset_name
        )
        score_path = os.path.join(
            self.dataset_data_path,
            'score',
        )
        graph_path = os.path.join(
            self.dataset_data_path,
            'graph',
        )

        self.barinel_score_name = os.path.join(
            score_path,
            train_release + '_barinel_normal.json',
        )
        self.dice_score_name = os.path.join(
            score_path,
            train_release + '_dice_normal.json',
        )
        self.dstar_score_name = os.path.join(
            score_path,
            train_release + '_dstar_normal.json',
        )
        self.ochiai_score_name = os.path.join(
            score_path,
            train_release + '_ochiai_normal.json',
        )
        self.op2_score_name = os.path.join(
            score_path,
            train_release + '_op2_normal.json',
        )
        self.tarantula_score_name = os.path.join(
            score_path,
            train_release + '_tarantula_normal.json',
        )

        self.graph_file_name = os.path.join(
            graph_path,
            train_release + '_SF_graph.txt',
        )
        self.barinel_graph_name = os.path.join(
            graph_path,
            train_release + '_barinel_SF_graph.txt',
        )
        self.op2_graph_name = os.path.join(
            graph_path,
            train_release + '_op2_SF_graph.txt',
        )
        self.dstar_graph_name = os.path.join(
            graph_path,
            train_release + '_dstar_SF_graph.txt',
        )
        self.ochiai_graph_name = os.path.join(
            graph_path,
            train_release + '_ochiai_SF_graph.txt',
        )
        self.tarantula_graph_name = os.path.join(
            graph_path,
            train_release + '_tarantula_SF_graph.txt',
        )

        self.dataset_root_path = (
            resolve_dataset_root_path(
                self.dataset_name
            )
        )
        self.file_level_dataset_path = os.path.join(
            self.dataset_root_path,
            self.dataset_name,
            'File-level',
        ) + os.sep
        self.line_level_dataset_path = os.path.join(
            self.dataset_root_path,
            self.dataset_name,
            'Line-level',
        ) + os.sep

        self.project_name = train_release.split('-')[0]
        np.random.seed(0)
        self.random_state = 0
        self.threshold_effort = 0.2

        self.train_release = train_release
        self.test_release = test_release

        self.vector = CountVectorizer(lowercase=False, min_df=2)
        self.clf = LogisticRegression(random_state=0)

        if is_realistic:
            self.train_text, self.train_text_lines, self.train_label, self.train_filename, self.train_text_lines_without_comments = read_file_level_dataset(
                train_release,
                file_path=str(
                    PROJECT_ROOT / 'Dataset' / 'File-level'
                ) + os.sep,
            )
        else:
            self.train_text, self.train_text_lines, self.train_label, self.train_filename, self.train_text_lines_without_comments = read_file_level_dataset(
                train_release,
                file_path=self.file_level_dataset_path,
            )

        self.test_text, self.test_text_lines, self.test_labels, self.test_filename, self.test_text_lines_without_comments = read_file_level_dataset(
            test_release,
            file_path=self.file_level_dataset_path,
        )

        self.file_level_result_file = f'{self.file_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.line_level_result_file = f'{self.line_level_result_path}{self.project_name}/{self.test_release}-result.csv'
        self.buggy_density_file = f'{self.buggy_density_path}{self.test_release}-density.csv'
        self.commit_buggy_path = str(
            PROJECT_ROOT
            / 'dataset'
            / 'Bug-Info'
            / self.test_release.split('-')[0]
        )

        self.init_file_path()

        self.test_pred_labels = []
        self.test_pred_scores = []
        self.test_pred_density = dict()

        self.oracle_line_dict, self.oracle_line_set = self.get_oracle_lines()
        self.predicted_buggy_lines = []
        self.predicted_buggy_score = []
        self.predicted_density = []

        self.num_total_lines = sum([len(lines) for lines in self.test_text_lines])
        self.num_total_lines_without_comments = sum([len(lines) for lines in self.test_text_lines_without_comments])
        self.num_actual_buggy_lines = len(self.oracle_line_set)

        if self.num_total_lines_without_comments != 0:
            self.line_threshold = self.num_actual_buggy_lines / self.num_total_lines_without_comments
        else:
            self.line_threshold = 0.0

        self.num_predict_buggy_lines = len(self.predicted_buggy_lines)

    def init_file_path(self):
        make_path(self.result_path)
        make_path(self.file_level_result_path)
        make_path(self.line_level_result_path)
        make_path(self.buggy_density_path)
        make_path(f'{self.file_level_result_path}{self.project_name}/')
        make_path(f'{self.line_level_result_path}{self.project_name}/')

    def get_oracle_lines(self):
        oracle_line_dict, oracle_line_list = (
            read_line_level_dataset_from_path(
                self.test_release,
                self.line_level_dataset_path,
            ),
            set(),
        )
        for file_name in oracle_line_dict:
            oracle_line_list.update(
                [
                    f'{file_name}:{line}'
                    for line in oracle_line_dict[file_name]
                ]
            )
        return oracle_line_dict, oracle_line_list

    def file_level_prediction(self):
        print(f"Prediction\t=>\t{self.test_release}")
        if USE_CACHE and os.path.exists(self.file_level_result_file):
            return

        train_vtr = self.vector.fit_transform(self.train_text)
        test_vtr = self.vector.transform(self.test_text)
        self.clf.fit(train_vtr, self.train_label)

        self.test_pred_labels = self.clf.predict(test_vtr)
        if self.model_name == 'MIT-TMI-SVM':
            self.test_pred_scores = np.array([score for score in self.test_pred_labels])
        else:
            self.test_pred_scores = np.array([score[1] for score in self.clf.predict_proba(test_vtr)])

        self.save_file_level_result()

    def line_level_prediction(self):
        print(f'Line level prediction for: {self.model_name}')
        pass

    def analyze_file_level_result(self):
        self.load_file_level_result()

        total_file, identified_file, total_line, identified_line, predicted_file, predicted_line, dropped_line = 0, 0, 0, 0, 0, 0, 0

        for index in range(len(self.test_labels)):
            buggy_line = len(self.test_text_lines[index])
            if self.test_pred_labels[index] == 1:
                predicted_file += 1
                predicted_line += buggy_line

        for index in range(len(self.test_labels)):
            if self.test_labels[index] == 1:
                buggy_line = len(self.oracle_line_dict[self.test_filename[index]])
                if self.test_pred_labels[index] == 1:
                    identified_line += buggy_line
                    identified_file += 1
                else:
                    dropped_line += len(self.oracle_line_dict[self.test_filename[index]])
                total_line += buggy_line
                total_file += 1

        print(f'Buggy file hit info: {identified_file}/{total_file} - {round(identified_file / total_file * 100, 1)}%')
        print(f'Buggy line hit info: {identified_line}/{total_line} - {round(identified_line / total_line * 100, 1)}%')
        print(f'Predicted {predicted_file} buggy files contain {predicted_line} lines')

        file_precision = metrics.precision_score(
            self.test_labels,
            self.test_pred_labels,
            zero_division=0,
        )
        file_recall = metrics.recall_score(
            self.test_labels,
            self.test_pred_labels,
            zero_division=0,
        )
        file_f1 = metrics.f1_score(
            self.test_labels,
            self.test_pred_labels,
            zero_division=0,
        )
        file_accuracy = metrics.accuracy_score(
            self.test_labels,
            self.test_pred_labels,
        )
        file_mcc = metrics.matthews_corrcoef(
            self.test_labels,
            self.test_pred_labels,
        )

        append_title = (
            not os.path.exists(
                self.file_level_evaluation_file
            )
        )
        title = (
            'release,precision,recall,f1-score,accuracy,mcc,'
            'identified/total files,max identified/total lines\n'
        )

        with open(
                self.file_level_evaluation_file,
                'a',
        ) as file:
            file.write(title) if append_title else None
            file.write(
                f'{self.test_release},'
                f'{file_precision},'
                f'{file_recall},'
                f'{file_f1},'
                f'{file_accuracy},'
                f'{file_mcc},'
                f'{identified_file}/{total_file},'
                f'{identified_line}/{total_line},'
                f'\n'
            )
        return

    def analyze_line_level_result(self):
        self.load_file_level_result()
        self.load_line_level_result()

        ranked_lines = self.build_ranked_predicted_buggy_lines()

        total_lines = self.num_total_lines_without_comments
        if len(ranked_lines) != total_lines:
            raise ValueError(
                'The final line ranking does not cover all evaluable '
                f'lines: {len(ranked_lines)} != {total_lines}.'
            )

        effort_20_count = int(
            total_lines * self.threshold_effort
        )
        inspected_lines = ranked_lines[:effort_20_count]

        tp = sum(
            1
            for line in inspected_lines
            if line in self.oracle_line_set
        )
        fp = effort_20_count - tp
        fn = self.num_actual_buggy_lines - tp
        tn = total_lines - tp - fp - fn

        print(
            f'Total lines: {total_lines}\n'
            f'Buggy lines: {self.num_actual_buggy_lines}\n'
            f'Ranked lines: {len(ranked_lines)}\n'
            f'Inspected lines at 20% effort: '
            f'{effort_20_count}\n'
            f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}'
        )

        precision = (
            0.0
            if tp + fp == 0
            else tp / (tp + fp)
        )
        recall = (
            0.0
            if tp + fn == 0
            else tp / (tp + fn)
        )
        far = (
            0.0
            if fp + tn == 0
            else fp / (fp + tn)
        )

        d2h = math.sqrt(
            (1.0 - recall) ** 2 + far ** 2
        ) / math.sqrt(2.0)

        mcc_denominator = (
                (tp + fp)
                * (tp + fn)
                * (tn + fp)
                * (tn + fn)
        )
        mcc = (
            0.0
            if mcc_denominator == 0
            else (
                    (tp * tn - fp * fn)
                    / math.sqrt(mcc_denominator)
            )
        )

        f1 = (
            0.0
            if precision + recall == 0
            else (
                    2.0
                    * precision
                    * recall
                    / (precision + recall)
            )
        )

        (
            ifa,
            effort_20,
            r_0,
            r_10,
            r_20,
            r_30,
            r_40,
            r_50,
            r_60,
            r_70,
            r_80,
            r_90,
            r_100,
        ) = self.get_rank_performance(ranked_lines)

        if not math.isclose(
                recall,
                r_20,
                rel_tol=1e-12,
                abs_tol=1e-12,
        ):
            raise ValueError(
                'Recall and Recall@20%LOC are inconsistent: '
                f'{recall} != {r_20}.'
            )

        labels = [
            1 if line in self.oracle_line_set else 0
            for line in ranked_lines
        ]

        if len(np.unique(labels)) == 2:
            total_ranked_lines = len(ranked_lines)
            ranking_scores = [
                (total_ranked_lines - index)
                / total_ranked_lines
                for index in range(total_ranked_lines)
            ]
            auc = calc_auc(labels, ranking_scores)
        else:
            auc = float('nan')

        append_title = not os.path.exists(
            self.line_level_evaluation_file
        )
        print(f'append_title is {append_title}')

        title = (
            'release,tp,fp,fn,tn,'
            'precision,recall,far,d2h,mcc,f1,auc,ifa,'
            'recall_0,recall_10,recall_20,recall_30,'
            'recall_40,recall_50,recall_60,recall_70,'
            'recall_80,recall_90,recall_100,'
            'effort@20%recall\n'
        )

        with open(
                self.line_level_evaluation_file,
                'a',
        ) as file:
            file.write(title) if append_title else None
            file.write(
                f'{self.test_release},'
                f'{tp},{fp},{fn},{tn},'
                f'{precision},{recall},{far},{d2h},'
                f'{mcc},{f1},{auc},{ifa},'
                f'{r_0},{r_10},{r_20},{r_30},'
                f'{r_40},{r_50},{r_60},{r_70},'
                f'{r_80},{r_90},{r_100},'
                f'{effort_20}\n'
            )

        return

    def build_evaluable_line_ids_by_file(self):

        evaluable_line_ids_by_file = {}

        for (
                filename,
                source_lines,
                lines_without_comments,
        ) in zip(
            self.test_filename,
            self.test_text_lines,
            self.test_text_lines_without_comments,
        ):
            filename = str(filename)

            line_ids = [
                f'{filename}:{line_index + 1}'
                for line_index, source_line in enumerate(source_lines)
                if not (
                        str(source_line).startswith('*')
                        or str(source_line).startswith('/')
                )
            ]

            expected_count = len(lines_without_comments)

            if len(line_ids) != expected_count:
                line_ids = []
                source_start_index = 0

                for evaluable_line in lines_without_comments:
                    matched_index = None

                    for source_index in range(
                            source_start_index,
                            len(source_lines),
                    ):
                        if (
                                str(source_lines[source_index])
                                == str(evaluable_line)
                        ):
                            matched_index = source_index
                            break

                    if matched_index is None:
                        raise ValueError(
                            'Cannot align a non-comment line with its '
                            f'original line number in file {filename}.'
                        )

                    line_ids.append(
                        f'{filename}:{matched_index + 1}'
                    )
                    source_start_index = matched_index + 1

            if len(line_ids) != expected_count:
                raise ValueError(
                    'The number of evaluable lines is inconsistent '
                    f'for file {filename}: '
                    f'{len(line_ids)} != {expected_count}.'
                )

            evaluable_line_ids_by_file[filename] = line_ids

        total_evaluable_lines = sum(
            len(line_ids)
            for line_ids in evaluable_line_ids_by_file.values()
        )

        if (
                total_evaluable_lines
                != self.num_total_lines_without_comments
        ):
            raise ValueError(
                'The reconstructed evaluable-line count is '
                'inconsistent with the dataset: '
                f'{total_evaluable_lines} != '
                f'{self.num_total_lines_without_comments}.'
            )

        return evaluable_line_ids_by_file

    def build_ranked_predicted_buggy_lines(self):

        if (
                len(self.predicted_buggy_lines)
                != len(self.predicted_buggy_score)
        ):
            raise ValueError(
                'The numbers of predicted lines and prediction scores '
                'are inconsistent.'
            )

        if len(self.test_pred_density) == 0:
            self.save_buggy_density_file()

        evaluable_line_ids_by_file = (
            self.build_evaluable_line_ids_by_file()
        )
        all_evaluable_line_ids = {
            line_id
            for line_ids in evaluable_line_ids_by_file.values()
            for line_id in line_ids
        }

        test_pred_density = np.asarray(
            [
                self.test_pred_density[str(filename)]
                for filename in self.test_filename
            ],
            dtype=float,
        )
        predicted_file_labels = np.asarray(
            self.test_pred_labels,
            dtype=int,
        )

        file_order = np.lexsort(
            (
                np.arange(len(self.test_filename)),
                -test_pred_density,
                -predicted_file_labels,
            )
        )

        predicted_scores_by_line = {}

        for line, score in zip(
                self.predicted_buggy_lines,
                self.predicted_buggy_score,
        ):
            line = str(line)

            if line not in all_evaluable_line_ids:
                raise ValueError(
                    'The line-level result contains a line that is not '
                    f'evaluable in the target release: {line}'
                )

            if line in predicted_scores_by_line:
                raise ValueError(
                    'The line-level result contains a duplicate line '
                    f'identifier: {line}'
                )

            predicted_scores_by_line[line] = float(score)

        ranked_lines = []

        for file_index in file_order:
            filename = str(
                self.test_filename[int(file_index)]
            )
            source_ordered_lines = (
                evaluable_line_ids_by_file[filename]
            )
            source_order_map = {
                line_id: source_index
                for source_index, line_id
                in enumerate(source_ordered_lines)
            }

            scored_lines = [
                line_id
                for line_id in source_ordered_lines
                if line_id in predicted_scores_by_line
            ]
            missing_lines = [
                line_id
                for line_id in source_ordered_lines
                if line_id not in predicted_scores_by_line
            ]

            scored_lines.sort(
                key=lambda line_id: (
                    -predicted_scores_by_line[line_id],
                    source_order_map[line_id],
                )
            )

            ranked_lines.extend(scored_lines)
            ranked_lines.extend(missing_lines)

        if len(ranked_lines) != len(set(ranked_lines)):
            raise ValueError(
                'The final line ranking contains duplicate line '
                'identifiers.'
            )

        if set(ranked_lines) != all_evaluable_line_ids:
            missing_line_ids = sorted(
                all_evaluable_line_ids - set(ranked_lines)
            )
            unexpected_line_ids = sorted(
                set(ranked_lines) - all_evaluable_line_ids
            )
            raise ValueError(
                'The final line ranking is incomplete. '
                f'Missing lines: {missing_line_ids[:10]}; '
                f'Unexpected lines: {unexpected_line_ids[:10]}.'
            )

        if (
                len(ranked_lines)
                != self.num_total_lines_without_comments
        ):
            raise ValueError(
                'The final line ranking does not cover all evaluable '
                f'lines: {len(ranked_lines)} != '
                f'{self.num_total_lines_without_comments}.'
            )

        return ranked_lines

    def rank_strategy(self):
        ranked_predicted_buggy_lines = (
            self.build_ranked_predicted_buggy_lines()
        )

        max_effort = int(
            self.num_total_lines_without_comments
            * self.threshold_effort
        )
        print(
            f'Ranked lines: '
            f'{len(ranked_predicted_buggy_lines)}, '
            f'Max effort: {max_effort}\n'
        )

        return self.get_rank_performance(
            ranked_predicted_buggy_lines
        )

    def get_rank_performance(
            self,
            ranked_predicted_buggy_lines,
    ):

        total_lines = self.num_total_lines_without_comments
        total_buggy_lines = self.num_actual_buggy_lines
        ranked_lines = list(ranked_predicted_buggy_lines)

        if total_lines <= 0:
            raise ValueError(
                'The number of evaluated source-code lines must be '
                'greater than zero.'
            )

        if len(ranked_lines) != total_lines:
            raise ValueError(
                'The ranking length does not match the number of '
                f'evaluable lines: {len(ranked_lines)} != '
                f'{total_lines}.'
            )

        if len(ranked_lines) != len(set(ranked_lines)):
            raise ValueError(
                'The ranking contains duplicate line identifiers.'
            )

        def calculate_recall_at_effort(effort_ratio):
            effort_count = int(
                total_lines * effort_ratio
            )

            if effort_ratio == 1.0:
                effort_count = total_lines

            identified_buggy_lines = sum(
                1
                for line in ranked_lines[:effort_count]
                if line in self.oracle_line_set
            )

            if total_buggy_lines == 0:
                return 0.0

            return (
                    identified_buggy_lines
                    / total_buggy_lines
            )

        recall_0 = 0.0
        recall_10 = calculate_recall_at_effort(0.1)
        recall_20 = calculate_recall_at_effort(0.2)
        recall_30 = calculate_recall_at_effort(0.3)
        recall_40 = calculate_recall_at_effort(0.4)
        recall_50 = calculate_recall_at_effort(0.5)
        recall_60 = calculate_recall_at_effort(0.6)
        recall_70 = calculate_recall_at_effort(0.7)
        recall_80 = calculate_recall_at_effort(0.8)
        recall_90 = calculate_recall_at_effort(0.9)
        recall_100 = calculate_recall_at_effort(1.0)

        positive_positions = [
            index
            for index, line in enumerate(ranked_lines)
            if line in self.oracle_line_set
        ]
        ifa = (
            positive_positions[0]
            if positive_positions
            else total_lines
        )

        if total_buggy_lines == 0:
            effort_20_recall = 0.0
        else:
            required_buggy_lines = max(
                1,
                int(math.ceil(
                    total_buggy_lines * 0.2
                )),
            )

            identified_buggy_lines = 0
            inspected_line_count = total_lines

            for index, line in enumerate(
                    ranked_lines,
                    start=1,
            ):
                if line in self.oracle_line_set:
                    identified_buggy_lines += 1

                if (
                        identified_buggy_lines
                        >= required_buggy_lines
                ):
                    inspected_line_count = index
                    break

            effort_20_recall = (
                    inspected_line_count
                    / total_lines
            )

        return (
            ifa,
            effort_20_recall,
            recall_0,
            recall_10,
            recall_20,
            recall_30,
            recall_40,
            recall_50,
            recall_60,
            recall_70,
            recall_80,
            recall_90,
            recall_100,
        )

    def save_file_level_result(self):
        data = {'filename': self.test_filename,
                'line_threshold': self.line_threshold,
                'oracle': self.test_labels,
                'predicted_label': self.test_pred_labels,
                'predicted_score': self.test_pred_scores}
        data = pd.DataFrame(data,
                            columns=['filename', 'line_threshold', 'oracle', 'predicted_label', 'predicted_score'])
        data.to_csv(self.file_level_result_file, index=False)

    def save_line_level_result(self):

        if (
                len(self.predicted_buggy_lines)
                != len(self.predicted_buggy_score)
        ):
            raise ValueError(
                'The numbers of predicted lines and prediction scores '
                'are inconsistent.'
            )

        file_line_map = {
            str(filename): lines
            for filename, lines in zip(
                self.test_filename,
                self.test_text_lines,
            )
        }

        predicted_code_lines = []

        for predicted_line in self.predicted_buggy_lines:
            predicted_line = str(predicted_line)

            try:
                filename, line_number_text = predicted_line.rsplit(
                    ':',
                    1,
                )
                line_number = int(line_number_text)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    'Invalid predicted line identifier: '
                    f'{predicted_line}'
                ) from error

            if filename not in file_line_map:
                raise ValueError(
                    'Cannot find the predicted source file in the '
                    f'test dataset: {filename}'
                )

            source_lines = file_line_map[filename]

            if line_number < 1 or line_number > len(source_lines):
                raise ValueError(
                    'Predicted line number is outside the source-file '
                    f'range: {predicted_line}'
                )

            predicted_code_lines.append(
                source_lines[line_number - 1]
            )

        data = {
            'predicted_buggy_lines': self.predicted_buggy_lines,
            'predicted_buggy_score': self.predicted_buggy_score,
            'code_line': predicted_code_lines,
        }

        result_dataframe = pd.DataFrame(
            data,
            columns=[
                'predicted_buggy_lines',
                'predicted_buggy_score',
                'code_line',
            ],
        )

        result_dataframe.to_csv(
            self.line_level_result_file,
            index=False,
            encoding='utf-8',
        )

    def save_buggy_density_file(self):
        dataframe = pd.read_csv(
            self.line_level_result_file
        )
        self.predicted_buggy_lines = list(
            dataframe['predicted_buggy_lines']
        )

        evaluable_line_ids_by_file = (
            self.build_evaluable_line_ids_by_file()
        )
        predicted_line_count_by_file = {
            str(filename): 0
            for filename in self.test_filename
        }

        for line in self.predicted_buggy_lines:
            line = str(line)

            try:
                filename, _ = line.rsplit(':', 1)
            except ValueError as error:
                raise ValueError(
                    f'Invalid predicted line identifier: {line}'
                ) from error

            if filename not in predicted_line_count_by_file:
                raise ValueError(
                    'Cannot map the predicted line to a test file: '
                    f'{line}'
                )

            predicted_line_count_by_file[filename] += 1

        buggy_density = {}

        for filename in self.test_filename:
            filename = str(filename)
            evaluable_line_count = len(
                evaluable_line_ids_by_file[filename]
            )

            if evaluable_line_count == 0:
                buggy_density[filename] = 0.0
            else:
                buggy_density[filename] = (
                        predicted_line_count_by_file[filename]
                        / evaluable_line_count
                )

        self.test_pred_density = buggy_density

        density_dataframe = pd.DataFrame(
            {
                'filename': list(buggy_density.keys()),
                'test_pred_density': list(
                    buggy_density.values()
                ),
            }
        )
        density_dataframe.to_csv(
            self.buggy_density_file,
            index=False,
        )

    def load_file_level_result(self):
        if len(self.test_pred_labels) == 0 or len(self.test_pred_scores) == 0:
            df = pd.read_csv(self.file_level_result_file)
            self.test_pred_labels = np.array(df['predicted_label'])
            self.test_pred_scores = np.array(df['predicted_score'])

    def load_line_level_result(self):
        if len(self.predicted_buggy_lines) == 0:
            df = pd.read_csv(self.line_level_result_file)
            self.predicted_buggy_lines = list(df['predicted_buggy_lines'])
            self.predicted_buggy_score = list(df['predicted_buggy_score'])
            self.num_predict_buggy_lines = len(self.predicted_buggy_lines)
