import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clear.src.my_utils.helper import (
    glance_projects,
    linedp_projects,
)


def process_csv(input_file, output_file):
    with open(
            input_file,
            'r',
            encoding='utf-8',
            errors='ignore',
    ) as file:
        lines = file.readlines()
        new_lines = []

        for line in lines:
            new_line = line.replace('"', '""')
            new_line_safe = new_line
            temp = new_line.split(',', 2)
            #print(temp[0])
            #print(temp[1])
            #print(temp[2].strip())

            if ',' in temp[2]:
                #print(temp[2])
                new_part_3 = (
                    '"'
                    + temp[2].strip()
                    + '"'
                )
                new_lines.append(
                    temp[0]
                    + ','
                    + temp[1]
                    + ','
                    + new_part_3
                )
            else:
                new_lines.append(new_line_safe)

    with open(
            output_file,
            'w',
            encoding='utf-8',
            errors='ignore',
    ) as file:
        for line in new_lines:
            print(line.strip(), file=file)


datasets_projects = {
    # 'glance_dataset': glance_projects,
    'linedp_dataset': linedp_projects,
}

for dataset_name, projects in datasets_projects.items():
    dataset_dir = (
        f'../../../../dataset/{dataset_name}/Line-level/'
    )
    out_dir = (
        f'../../../../dataset/{dataset_name}/processed/'
        'Line-level/'
    )

    os.makedirs(out_dir, exist_ok=True)

    for project_name, releases in projects.items():
        for release in releases:
            process_csv(
                dataset_dir
                + release
                + '_defective_lines_dataset.csv',
                out_dir
                + release
                + '_defective_lines_dataset.csv',
            )
            print(
                f'[{dataset_name}] {release} done'
            )
