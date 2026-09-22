import json
import os

import pandas as pd

from baselines.sound.src.utils.config import (
    GLANCE_PROJECT_RELEASE_LIST,
    LINEDP_PROJECT_RELEASE_LIST,
)

DATASET_RELEASES = {
    # 'glance_dataset': GLANCE_PROJECT_RELEASE_LIST,
    'linedp_dataset': LINEDP_PROJECT_RELEASE_LIST,
}


def select_top100(dataset_name, release):
    data_root = f'../Data/{dataset_name}'

    json_file_name = data_root + '/score/' + release + '_barinel_normal.json'
    op2_file_name = data_root + '/score/' + release + '_op2_normal.json'
    dstar_file_name = data_root + '/score/' + release + '_dstar_normal.json'
    tarantula_file_name = data_root + '/score/' + release + '_tarantula_normal.json'
    ochiai_file_name = data_root + '/score/' + release + '_ochiai_normal.json'

    cf_json_file_name = data_root + '/n_score/' + release + '_cf.json'

    tokens_file_name = data_root + '/tokens/' + release + '_tokens.csv'

    selected_file_name = data_root + '/selected/' + release + '_barinel.csv'
    op2_selected_file_name = data_root + '/selected/' + release + '_op2.csv'
    dstar_selected_file_name = data_root + '/selected/' + release + '_dstar.csv'
    tarantula_selected_file_name = data_root + '/selected/' + release + '_tarantula.csv'
    ochiai_selected_file_name = data_root + '/selected/' + release + '_ochiai.csv'

    with open(json_file_name, 'r') as f:
        token_score = json.load(f)
    with open(cf_json_file_name, 'r') as f:
        cf_times = json.load(f)
    with open(op2_file_name, 'r') as f:
        op2_score = json.load(f)
    with open(dstar_file_name, 'r') as f:
        dstar_score = json.load(f)
    with open(tarantula_file_name, 'r') as f:
        tarantula_score = json.load(f)
    with open(ochiai_file_name, 'r') as f:
        ochiai_score = json.load(f)

    df = pd.read_csv(tokens_file_name)

    sorted_keys = sorted(
        token_score.keys(),
        key=lambda k: (
            token_score[k],
            cf_times.get(k, float('inf')),
        ),
        reverse=True,
    )
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    os.makedirs(os.path.dirname(selected_file_name), exist_ok=True)
    selected_data.to_csv(selected_file_name, index=False)

    sorted_keys = sorted(
        dstar_score.keys(),
        key=lambda k: (
            dstar_score[k],
            cf_times.get(k, float('inf')),
        ),
        reverse=True,
    )
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(dstar_selected_file_name, index=False)

    sorted_keys = sorted(
        tarantula_score.keys(),
        key=lambda k: (
            tarantula_score[k],
            cf_times.get(k, float('inf')),
        ),
        reverse=True,
    )
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(tarantula_selected_file_name, index=False)

    sorted_keys = sorted(
        ochiai_score.keys(),
        key=lambda k: (
            ochiai_score[k],
            cf_times.get(k, float('inf')),
        ),
        reverse=True,
    )
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(ochiai_selected_file_name, index=False)

    sorted_keys = sorted(
        op2_score.keys(),
        key=lambda k: (
            op2_score[k],
            cf_times.get(k, float('inf')),
        ),
        reverse=True,
    )
    keys_set = set(sorted_keys[:100])
    my_set = keys_set
    my_set.add('Bug')

    selected_columns = [col for col in df.columns if col in my_set]
    selected_data = df[selected_columns]
    selected_data.to_csv(op2_selected_file_name, index=False)


for dataset_name, releases in DATASET_RELEASES.items():
    for release in releases:
        select_top100(dataset_name, release)
