import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from gensim.models import Word2Vec

import more_itertools

from DeepLineDP_model import *
from baselines.sound.src.utils.config_for_deeplinedp import *
from clear.src.my_utils.helper import linedp_projects

DATASETS_PROJECTS = {
    #'glance_dataset': all_releases,
     'linedp_dataset': linedp_projects,
}

PREPROCESSED_DATA_ROOT = '../../../../dataset'


def get_dataset_df(which_dataset, release):
    data_path = os.path.join(
        PREPROCESSED_DATA_ROOT,
        which_dataset,
        'preprocessed_data',
        release + '.csv',
    )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f'Preprocessed data file not found: {data_path}'
        )

    dataframe = pd.read_csv(data_path)
    dataframe = dataframe.fillna('')
    dataframe = dataframe[dataframe['is_blank'] == False]
    dataframe = dataframe[dataframe['is_test_file'] == False]

    return dataframe


def train_word2vec_model(
        which_dataset,
        dataset_name,
        embedding_dim=50,
):
    w2v_path = os.path.join(
        get_w2v_path(),
        which_dataset,
    )

    if not os.path.exists(w2v_path):
        os.makedirs(w2v_path)

    releases = DATASETS_PROJECTS[which_dataset][dataset_name]

    for i, release in enumerate(releases[:-1]):
        save_path = f"{w2v_path}/{release}-{embedding_dim}dim.bin"

        if os.path.exists(save_path):
            print(f"Word2Vec model at {save_path} already exists")
            continue

        train_df = get_dataset_df(which_dataset, release)
        train_code_3d, _ = get_code3d_and_label(train_df, True)
        all_texts = list(more_itertools.collapse(train_code_3d[:], levels=1))

        word2vec = Word2Vec(
            all_texts,
            vector_size=embedding_dim,
            min_count=1,
            sorted_vocab=1
        )
        word2vec.save(save_path)
        print(f"Saved Word2Vec model at path {save_path}")


if __name__ == "__main__":
    for which_dataset, projects in DATASETS_PROJECTS.items():
        for dataset_name in projects:
            train_word2vec_model(
                which_dataset,
                dataset_name,
                50,
            )
