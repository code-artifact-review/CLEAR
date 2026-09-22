import os

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["JAX_PLATFORMS"] = "cuda"

from baselines.sound.src.utils.config import LINEDP_PROJECT_RELEASE_LIST, GLANCE_PROJECT_RELEASE_LIST

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from typing import *
from dibs.utils import visualize_ground_truth
from dibs.models import ErdosReniDAGDistribution, BGe, ScaleFreeDAGDistribution, DenseNonlinearGaussian, LinearGaussian
from dibs.inference import MarginalDiBS, JointDiBS
from dibs.graph_utils import elwise_acyclic_constr_nograd
from jax.scipy.special import logsumexp

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"

ENDOGENOUS_NODES = ['Bug']

thd = 0.0
DATA_ROOT = "../Data"
data_path = ""
result_path = ""

glance_releases = GLANCE_PROJECT_RELEASE_LIST

linedp_releases = LINEDP_PROJECT_RELEASE_LIST

DATASET_RELEASES = {
    #'glance_dataset': glance_releases,
    'linedp_dataset': linedp_releases,
}

collected_df = pd.DataFrame()


def read_data(folder: str, selected_name: str) -> pd.DataFrame:
    df = pd.DataFrame()
    for dir_path, _, file_names in os.walk(folder):
        for file_name in file_names:
            if file_name == selected_name:
                file_path = os.path.join(folder, file_name)
                df = pd.read_csv(file_path)
                print(f"Read {file_name} with {df.shape[0]} rows")
    return df


def matrix_to_dgraph(matrix: np.ndarray, columns: List[str], threshold: float = 1.0) -> List[str]:
    dgraph = []
    for i in range(matrix.shape[0]):
        if matrix[i, collected_df.shape[1] - 1] > threshold:
            dgraph.append(
                f"{columns[i]} -> {columns[collected_df.shape[1] - 1]} :{matrix[i, collected_df.shape[1] - 1]}")
    return dgraph


def compute_expected_graph(*, dist):
    n_vars = dist.g.shape[1]

    is_dag = elwise_acyclic_constr_nograd(dist.g, n_vars) == 0
    assert is_dag.sum() > 0, "No acyclic graphs found"

    particles = dist.g[is_dag, :, :]
    log_weights = dist.logp[is_dag] - logsumexp(dist.logp[is_dag])

    expected_g = jnp.zeros_like(particles[0])
    for i in range(particles.shape[0]):
        expected_g += jnp.exp(log_weights[i]) * particles[i, :, :]

    return expected_g


def discover_barinel(release):
    selected_name = release + '_barinel.csv'
    graph_file_name = release + '_barinel_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(
        collected_data.shape[1],
        n_edges_per_node=2
    )
    model = BGe(n_vars=collected_data.shape[1])
    dibs = MarginalDiBS(
        x=collected_data,
        interv_mask=interv_mask,
        graph_model=model_graph,
        likelihood_model=model
    )

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path + graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_op2(release):
    selected_name = release + '_op2.csv'
    graph_file_name = release + '_op2_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(
        collected_data.shape[1],
        n_edges_per_node=2
    )
    model = BGe(n_vars=collected_data.shape[1])
    dibs = MarginalDiBS(
        x=collected_data,
        interv_mask=interv_mask,
        graph_model=model_graph,
        likelihood_model=model
    )

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path + graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_dstar(release):
    selected_name = release + '_dstar.csv'
    graph_file_name = release + '_dstar_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(
        collected_data.shape[1],
        n_edges_per_node=2
    )
    model = BGe(n_vars=collected_data.shape[1])
    dibs = MarginalDiBS(
        x=collected_data,
        interv_mask=interv_mask,
        graph_model=model_graph,
        likelihood_model=model
    )

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path + graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_ochiai(release):
    selected_name = release + '_ochiai.csv'
    graph_file_name = release + '_ochiai_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(
        collected_data.shape[1],
        n_edges_per_node=2
    )
    model = BGe(n_vars=collected_data.shape[1])
    dibs = MarginalDiBS(
        x=collected_data,
        interv_mask=interv_mask,
        graph_model=model_graph,
        likelihood_model=model
    )

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path + graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


def discover_tarantula(release):
    selected_name = release + '_tarantula.csv'
    graph_file_name = release + '_tarantula_SF_graph.txt'
    rand_key = jax.random.PRNGKey(0)

    collected_df = read_data(data_path, selected_name)
    collected_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    collected_df.dropna(inplace=True)
    collected_df = collected_df.sample(frac=1).reset_index(drop=True)

    print(f"Collected data shape: {collected_df.shape}")
    print(f"Collected data columns: {collected_df.columns}")

    interv_df = collected_df.copy()
    for col in interv_df.columns:
        if col in ENDOGENOUS_NODES:
            interv_df[col] = 0
    for col in interv_df.columns:
        if col not in ENDOGENOUS_NODES:
            interv_df[col] = 1
    interv_mask = interv_df.values
    interv_mask[interv_mask > 0] = 1
    interv_mask = interv_mask.astype(int)
    interv_mask = jnp.array(interv_mask)

    scaler = StandardScaler()
    collected_data = scaler.fit_transform(collected_df)

    model_graph = ScaleFreeDAGDistribution(
        collected_data.shape[1],
        n_edges_per_node=2
    )
    model = BGe(n_vars=collected_data.shape[1])
    dibs = MarginalDiBS(
        x=collected_data,
        interv_mask=interv_mask,
        graph_model=model_graph,
        likelihood_model=model
    )

    rand_key, subk = jax.random.split(rand_key)
    gs = dibs.sample(key=subk, n_particles=10, steps=600, callback_every=600, callback=None)

    print(f"dibs sample is finished")

    dibs_output = dibs.get_mixture(gs)
    expected_g = compute_expected_graph(dist=dibs_output)

    visualize_ground_truth(jnp.array(expected_g))
    dgraph = matrix_to_dgraph(expected_g, collected_df.columns, threshold=thd)
    f = open(result_path + graph_file_name, 'w')
    print(len(dgraph), end='\n', file=f)
    for line in dgraph:
        print(line, end='\n', file=f)
    f.close()


discovery_functions = [
    ("Barinel", discover_barinel),
    ("OP2", discover_op2),
    ("DStar", discover_dstar),
    ("Ochiai", discover_ochiai),
    ("Tarantula", discover_tarantula),
]


def run_dataset(dataset_name, releases):

    global data_path
    global result_path

    data_path = os.path.join(
        DATA_ROOT,
        dataset_name,
        'selected',
    )
    result_path = os.path.join(
        DATA_ROOT,
        dataset_name,
        'graph',
    ) + os.sep

    os.makedirs(
        result_path,
        exist_ok=True,
    )

    for release in tqdm(
            releases,
            desc=f'Processing {dataset_name} releases',
            unit='release',
            dynamic_ncols=True,
    ):
        for method_name, discovery_function in tqdm(
                discovery_functions,
                desc=f'[{dataset_name}][{release}] Discovering graphs',
                unit='method',
                leave=False,
                dynamic_ncols=True,
        ):
            discovery_function(release)


if __name__ == '__main__':
    for dataset_name, releases in DATASET_RELEASES.items():
        run_dataset(
            dataset_name,
            releases,
        )
