#!/usr/bin/env python3

import dgl
import torch
from tqdm import tqdm
from dataset import GraphDataset
from dgl.dataloading import GraphDataLoader
def to_bidirected_save_edge_feature(g, copy_ndata):
    ret_g = dgl.add_reverse_edges(g, copy_ndata=copy_ndata, copy_edata=True)
    ret_g = dgl.to_simple(g, return_counts=None, copy_ndata=copy_ndata, copy_edata=True)
    return ret_g

def load_dgl_data(dataset, version=None, cv_fold=None, homo=False, bidirected=False, timestamp=True):
    assert dataset in ['wget', 'streamspot'], 'Invalid dataset.'

    data_path = f'../data/{dataset}'

    if dataset == 'wget':
        if not version.startswith('_'):
            version = f'_{version}'
        dataset += f'{version}'

    graph_path = f'{data_path}/{dataset}_dgl_graphs_homo.bin' if homo else f'{data_path}/{dataset}_dgl_graphs.bin'

    graphs, graph_attr = dgl.load_graphs(graph_path)
    with open(f'{data_path}/{dataset}_relations.txt') as f:
        relations = [line.strip() for line in f]

    feat_dim = graphs[0].ndata['feat'].size(-1)
    # print(graphs[0].edata['timestamp'])
    if bidirected and not timestamp:
        graphs = [
            dgl.to_bidirected(g, copy_ndata=True)
            for g in graphs
        ]
    if bidirected and timestamp:
        graphs = [to_bidirected_save_edge_feature(g, copy_ndata=True)
                for g in graphs
        ]
    # print(graphs[0].edata['timestamp'])
    if cv_fold is None:
        train_dataset = GraphDataset(dataset, graphs, graph_attr['label'].float())
        test_dataset = None
    
    else:
        # TODO: How to best index a python list with boolean mask?
        #  train_mask = (graph_attr['cv_fold'] != cv_fold)
        train_graphs, test_graphs, val_graphs = [], [], []
        train_labels, test_labels, val_labels = [], [], []
        max_fold = 5
        for g, cv, lbl in zip(graphs, graph_attr['cv_fold'], graph_attr['label']):
            if cv == cv_fold:
                test_graphs.append(g)
                test_labels.append(lbl)
            # if cv+1 % max_fold == cv_fold:
            #     val_graphs.append(g)
            #     val_labels.append(lbl)
            else:
                print(cv)
                train_graphs.append(g)
                train_labels.append(lbl)
        print(len(train_graphs))
        print(len(test_graphs))
        train_dataset = GraphDataset(dataset, train_graphs, torch.FloatTensor(train_labels))
        test_dataset = GraphDataset(dataset, test_graphs, torch.FloatTensor(test_labels))
        val_dataset = GraphDataset(dataset, val_graphs, torch.FloatTensor(val_labels))

    return train_dataset, test_dataset, val_dataset, feat_dim, relations

# (train_dataset,
#          test_dataset,
#          feat_dim,
#          relations) = load_dgl_data("streamspot", None, cv_fold=4, homo=None, bidirected=True)

# train_loader = GraphDataLoader(
#             train_dataset,
#             batch_size=1,
#             shuffle=True,
#             num_workers=1,
#             pin_memory=True
#         )
# data_iter = tqdm(
#                 train_loader,
#                 desc=f'Epoch: {1:02}',
#                 total=len(train_loader),
#                 position=0
#             )
# label1 = 0
# label0 = 0
# for i, (batch_graph, labels) in enumerate(data_iter):
#     if int(labels[0]) == 1:
#         label1 += 1
#     else:
#         label0 += 1
# print(label0, label1)