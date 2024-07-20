import os
import sys
import time

import ogb
from torch.optim import optimizer

print(sys.path)

sys.path.append('/home/ubuntu/graph-reordering')

import graphlearn_torch as glt
from training.pyg.full_graph_training import full_graph_training_pyg  # NOQA
from training.pyg.mini_batch_training import mini_batch_training_pyg  # NOQA
from models.Models import Models  # NOQA
from utils.timer import Timer  # NOQA
from utils.logger import Logger  # NOQA
from dgl.data.utils import load_graphs  # NOQA
import dgl.nn as dglnn  # NOQA
import torch_geometric  # NOQA
from torch_geometric.data import Data  # NOQA
import torch.nn as nn  # NOQA
import torch  # NOQA

import torch.nn.functional as F
import numpy as np  # NOQA
from dgl.data.utils import save_graphs  # NOQA
import dgl  # NOQA
from pyarrow import csv  # NOQA
import pyarrow  # NOQA
import argparse  # NOQA
import json  # NOQA
from tqdm import tqdm
from ogb.nodeproppred import Evaluator
from torch_geometric.loader import NeighborSampler


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", required=True,
                        type=str, help="graphsage, gcn or gat")
    parser.add_argument("-reordering_strategy", required=True,
                        type=str, help="rabbit, gorder, metis-65536, rand-0")
    parser.add_argument("-neighbors_per_layer", required=True,
                        type=int, nargs='+', help="The number of neighbors to sample per layer in NeighborSampler")
    parser.add_argument("-batch_size", required=False,
                        type=int, help="The number of target vertices to sample.", default=1024)
    parser.add_argument("-num_epochs", required=True,
                        type=int, help="The number of epochs to train.", default=8)
    parser.add_argument("-num_features", required=True,
                        type=int, help="The number of features.", default=16)
    parser.add_argument("-num_layers", required=True,
                        type=int, help="The number of layers.", default=2)
    parser.add_argument("-hidden_dim", required=True,
                        type=int, help="The number of hidden dimension.", default=16)
    parser.add_argument("-path_to_result_metrics", required=True,
                        type=str,
                        help="in this file we store the results, e.g, /media/akshaya/Local Disk1/reorder-training/experiments/83737383828293.json")

    return parser


def train(model, epoch, data, train_loader):
    model.train()
    pbar = tqdm(total=data.split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    step = 0
    data.node_labels = data.node_labels.to("cpu")
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to("cpu") for adj in adjs]
        optimizer.zero_grad()
        out = model(data.node_features[n_id], adjs)
        loss = F.nll_loss(out, data.node_labels[n_id[:batch_size]])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(data.node_labels[n_id[:batch_size]]).sum())
        step += 1
        pbar.update(batch_size)

    pbar.close()

    loss = total_loss / step
    approx_acc = total_correct / data.split_idx['train'].size(0)
    return loss, approx_acc


def test(model, data):
    evaluator = Evaluator(name='ogbn-products')
    model.eval()
    out = model.inference(data.node_features)

    y_true = data.node_labels.cpu().unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.split_idx['train']],
        'y_pred': y_pred[data.split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[data.split_idx['valid']],
        'y_pred': y_pred[data.split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.split_idx['test']],
        'y_pred': y_pred[data.split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc


def mini_batch_training(model, config, data, timer, train_loader):

    test_accs = []
    for run in range(1, 3):   # No.of Runs
        print('')
        print(f'Run {run:02d}:')
        print('')

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        best_val_acc = final_test_acc = 0
        for epoch in range(1, config['num_epochs'] + 1):
            epoch_start = time.time()
            loss, acc = train(model, epoch, data, train_loader)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}',
                  f'Epoch Time: {time.time() - epoch_start}')

            if epoch > 5:
                train_acc, val_acc, test_acc = test(model, data)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    final_test_acc = test_acc
        test_accs.append(final_test_acc)

    test_acc = torch.tensor(test_accs)
    print('============================')
    print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')


def run_training(graph, config, timer=None):
    in_size = config["num_features"]
    out_size = 10

    N_LAYERS = config["num_layers"]
    HIDDEN_DIM = config["hidden_dim"]

    # PyG NeighborSampler
    test_loader = NeighborSampler(graph.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=config["batch_size"], shuffle=False, num_workers=4)

    model = Models.get_models()[config["model"]](in_size, out_size, N_LAYERS, HIDDEN_DIM, test_loader)
    model = model.to("cpu")
    timer.start("Training Start")

    # graphlearn_torch NeighborLoader
    train_loader = glt.loader.NeighborLoader(graph,
                                             [15, 10, 5],  # [25, 10, 5]
                                             graph.split_idx['train'],
                                             batch_size=1024,
                                             shuffle=True,
                                             drop_last=True,
                                             device=torch.device('cpu'),
                                             as_pyg_v1=True)
    mini_batch_training(model=model, config=config, data=graph, timer=timer, loader=train_loader)
    timer.stop("Training Stop")
    return


def measure_training_time(graph, config):
    timer = Timer()
    run_training(graph, config, timer=timer)
    return timer.get_all()


def load_pyg_graph(path, config):
    """
    Load a PyG graph from a given path.

    Args:
        path (str): The path to the saved PyG graph.
        config (dict): The configuration settings.

    Returns:
        torch_geometric.data.Data: The loaded PyG graph.
    """
    data = torch.load(path)
    data = data.to(torch.device("cpu"))
    return data


def adapt_graph_feature_size_pyg(g, config):
    """
    Adapts the graph feature size by replacing the existing features with new ones.

    Args:
        g (torch_geometric.data.Data): The input graph.
        config (dict): Configuration parameters.

    Returns:
        torch_geometric.data.Data: The modified graph with updated features.
    """
    NUM_FEATURES = config["num_features"]
    new_features = torch.ones(g.num_nodes, NUM_FEATURES).to(torch.device("cpu"))
    g.x = new_features
    return g


def write_results(config, durations):
    """
    Write the experiment results to a file.

    Args:
        config (dict): The configuration dictionary.
        durations (dict): The dictionary containing the durations and experiment configuration.

    Returns:
        None
    """
    for k, v in durations.items():
        config[k] = v
    with open(config["path_to_result_metrics"], "w") as f:
        json.dump(config, f)
    print(f"The result of the experiment is: {config}\n\n")


def run_pyg_training(config):
    """
    Runs the PyTorch Geometric training experiment.

    Args:
        config (dict): Configuration parameters for the experiment.

    Returns:
        None
    """
    path_to_graph = "data/ogbn-products-{}.pt".format(config["reordering_strategy"])
    data = load_pyg_graph(path_to_graph, config)
    data = adapt_graph_feature_size_pyg(data, config)
    # GLT CODE STARTS
    glt_dataset = glt.data.Dataset()
    glt_dataset.init_graph(
        edge_index=data[0].edge_index,
        graph_mode='CPU',
        directed=False
    )
    glt_dataset.init_node_features(
        node_feature_data=data.x,
        sort_func=glt.data.sort_by_in_degree,
        split_ratio=0.2,
        device_group_list=[glt.data.DeviceGroup(0, [0])],
        with_gpu=False
    )
    glt_dataset.init_node_labels(node_label_data=data.y)
    #  GLT CODE ENDS
    durations, means, variances = measure_training_time(glt_dataset, config)
    write_results(config=config, durations=durations)


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()

    conf = vars(args)
    conf_copy = vars(args)

    print(f"\nWe are starting this experiment: {conf}")

    with open(conf["path_to_result_metrics"], "w") as f:
        conf_copy["failed"] = "start"
        json.dump(conf_copy, f)
        run_pyg_training(config=conf)
