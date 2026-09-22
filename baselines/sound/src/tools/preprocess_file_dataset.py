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

        for i, line in enumerate(lines):
            if '.java,' not in line and i < len(lines) - 1:
                if '.java,' not in lines[i + 1]:
                    new_lines.append(
                        line.replace('"', '""')
                    )
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

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
        f'../../../../dataset/{dataset_name}/File-level/'
    )
    out_dir = (
        f'../../../../dataset/{dataset_name}/processed/'
        'File-level/'
    )

    os.makedirs(out_dir, exist_ok=True)

    for project_name, releases in projects.items():
        for release in releases:
            process_csv(
                dataset_dir
                + release
                + '_ground-truth-files_dataset.csv',
                out_dir
                + release
                + '_ground-truth-files_dataset.csv',
            )
            print(
                f'[{dataset_name}] {release} done'
            )
