# from models.glance import *
# from models.linedp import *
from baselines.sound.src.models.glance import *
from baselines.sound.src.models.linedp import *
from baselines.sound.src.utils.config import (
    GLANCE_PROJECT_RELEASE_LIST,
    LINEDP_PROJECT_RELEASE_LIST,
)
from models.mymodel import *

import warnings

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings('ignore')
warnings.simplefilter(
    action='ignore',
    category=FutureWarning,
)

DATASET_RELEASE_LISTS = {
    #'glance_dataset': GLANCE_PROJECT_RELEASE_LIST,
    'linedp_dataset': LINEDP_PROJECT_RELEASE_LIST,
}


def get_project_releases_dict(release_list):

    project_releases_dict = {}

    for release in release_list:
        project = release.split('-')[0]

        if project not in project_releases_dict:
            project_releases_dict[project] = [release]
        else:
            project_releases_dict[project].append(release)

    return project_releases_dict


def run_cross_release_predict(
        dataset_name,
        release_list,
        prediction_model,
):

    project_releases_dict = get_project_releases_dict(
        release_list
    )

    for project, releases in project_releases_dict.items():
        for release_index in range(
                len(releases) - 1
        ):
            train_release = releases[release_index]
            test_release = releases[
                release_index + 1
                ]

            print(
                f'========== [{dataset_name}] '
                f'{prediction_model.model_name} '
                f'CR PREDICTION for {test_release} '
                '================'
            )

            model = prediction_model(
                train_release,
                test_release,
            )

            model.file_level_prediction()
            model.analyze_file_level_result()
            model.line_level_prediction()
            model.analyze_line_level_result()


def run_models_for_dataset(
        dataset_name,
        release_list,
):

    # GLANCE
    run_cross_release_predict(
        dataset_name,
        release_list,
        Glance_LR_Mixed_Sort,
    )
    run_cross_release_predict(
        dataset_name,
        release_list,
        Glance_EA_Mixed_Sort,
    )
    run_cross_release_predict(
        dataset_name,
        release_list,
        Glance_MD_Mixed_Sort,
    )

    # LineDP
    run_cross_release_predict(
        dataset_name,
        release_list,
        LineDP_mixedsort,
    )

    # SOUND
    run_cross_release_predict(
        dataset_name,
        release_list,
        Barinel,
    )
    run_cross_release_predict(
        dataset_name,
        release_list,
        Dstar,
    )
    run_cross_release_predict(
        dataset_name,
        release_list,
        Ochiai,
    )
    run_cross_release_predict(
        dataset_name,
        release_list,
        Op2,
    )
    run_cross_release_predict(
        dataset_name,
        release_list,
        Tarantula,
    )


def run_default():

    for (
            dataset_name,
            release_list,
    ) in DATASET_RELEASE_LISTS.items():
        print(
            f'\n========== START DATASET: '
            f'{dataset_name} ==========\n'
        )

        run_models_for_dataset(
            dataset_name,
            release_list,
        )

        print(
            f'\n========== FINISH DATASET: '
            f'{dataset_name} ==========\n'
        )


if __name__ == '__main__':
    run_default()
