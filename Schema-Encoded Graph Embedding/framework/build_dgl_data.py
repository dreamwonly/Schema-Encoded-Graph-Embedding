#!/usr/bin/env python3
from collections import defaultdict
import os
import argparse
import json
from random import randint
from tkinter import E
import torch
import dgl
import re 
import numpy as np
import random
def map(gid):
    if gid >= 0 and gid <= 99:
        graph_name = "youtube"
        label = 0
    elif gid >= 100 and gid <= 199:
        graph_name = "gmail"
        label = 0
    elif gid >= 200 and gid <= 299:
        graph_name = "vgame"
        label = 0
    elif gid >= 300 and gid <= 399:
        graph_name = "attack"
        label = 1
    elif gid >= 400 and gid <= 499:
        graph_name = "download"
        label = 0
    elif gid >= 500 and gid <= 599:
        graph_name = "cnn"
        label = 0
    return graph_name, label, gid

def main(args):
    dataset = args.dataset.lower()
    data_path = f'../data/{dataset}'
    stream_data_path = f'../data/wget'
    args.timestamp = False

    if dataset == 'wget':
        dataset += f'_{args.attack}'
        dirs_with_label = {f'{data_path}/benign': 0}
        if args.attack == 'sc1' or args.attack == 'both':
            dirs_with_label.update({f'{data_path}/attack_sc1': 1})
        if args.attack == 'sc2' or args.attack == 'both':
            dirs_with_label.update({f'{data_path}/attack_sc2': 1})

    elif dataset == 'streamspot':
        dirs_with_label = {
            f'{data_path}/cnn': 0,
            f'{data_path}/download': 0,
            f'{data_path}/gmail': 0,
            f'{data_path}/vgame': 0,
            f'{data_path}/youtube': 0,
            f'{data_path}/attack': 1,
        }
    else:
        raise ValueError('Invalid dataset.')

    graphs, labels, cv_folds = [], [], []
    relations = set()
    if not args.timestamp and 'wget' not in dataset:
        for graph_dir, label in dirs_with_label.items():
            for graph_fname in os.listdir(graph_dir):
                edge_timestamp = dict()
                pattern = re.compile(r"\d+")
                graph_name, label, gid = map(int(pattern.findall(graph_fname)[0]))
                print(graph_name, gid)
                with open(os.path.join(stream_data_path, graph_name + "_data", "base_train", "base-{}-v2-{}.txt".format(graph_name, gid)), "r") as f:
                    for line in f:
                        rel = line.strip().split(" ")[2].split(":")[2]
                        src_node_str = line.strip().split(" ")[0]
                        dst_node_str = line.strip().split(" ")[1]
                        # print(src_node_str, rel, dst_node_str)
                        edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[3])
                with open(os.path.join(stream_data_path, graph_name + "_data", "stream_train", "stream-{}-v2-{}.txt".format(graph_name, gid)), "r") as f:
                    for line in f:
                        rel = line.strip().split(" ")[2].split(":")[2]
                        src_node_str = line.strip().split(" ")[0]
                        dst_node_str = line.strip().split(" ")[1]
                        edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[5])
                with open(os.path.join(graph_dir, graph_fname)) as f:
                    graph_json = json.load(f)
                # print(edge_timestamp)
                graph_data, node_id_map = {}, {}
                time_feature_rel_dict = dict()
                i = 0
                for k, v in graph_json.items():
                    rel = k.split('-')[1]
                    relations.add(rel)

                    edge_schema = (dgl.NTYPE, rel, dgl.NTYPE)

                    src, dst = v
                    if 'wget' in dataset:
                        src_idx, dst_idx = src, dst
                    else:
                        src_idx, dst_idx = [], []
                        edge_timestamp_list = []
                        for node_str in src:
                            if node_str in node_id_map:
                                node = node_id_map[node_str]
                            else:
                                node_id_map[node_str] = node = i
                                i += 1

                            src_idx.append(node)

                        for node_str in dst:
                            if node_str in node_id_map:
                                node = node_id_map[node_str]
                            else:
                                node_id_map[node_str] = node = i
                                i += 1

                            dst_idx.append(node)
                        for i in range(len(src)):
                            m, s = divmod(edge_timestamp[(src[i], rel, dst[i])], 60)
                            h, m = divmod(m, 60)
                            h_m_s = [[h], [m], [s]]
                            edge_timestamp_list.append(h_m_s)

                    graph_data[edge_schema] = (src_idx, dst_idx)
                    time_feature = np.array(edge_timestamp_list)
                    print(torch.from_numpy(time_feature))
                    time_feature_rel_dict[edge_schema] = torch.from_numpy(time_feature)
                # print(graph_data)
                tmp_g = dgl.heterograph(graph_data)
                tmp_g.edata['timestamp'] = time_feature_rel_dict
                tmp_g = dgl.to_simple(tmp_g, copy_edata=True)

                graphs.append(tmp_g)
                labels.append(label)
                # cv_folds.append(randint(0, 4))
    if 'wget' in dataset:
        graphs, labels, cv_folds = [], [], []
        relations = set()
        label = 0
        for gid in range(120):
            schemas = dict()
            edge_timestamp = dict()
            with open(os.path.join(stream_data_path, "benign", "base", "base-wget-{}.txt".format(gid)), "r") as f:
                for line in f:
                    rel = line.strip().split(" ")[2].split(":")[2]
                    src_node_str = line.strip().split(" ")[0]
                    dst_node_str = line.strip().split(" ")[1]
                    if rel in schemas:
                        schemas[rel].append((src_node_str, dst_node_str))
                    else:
                        schemas[rel] = [(src_node_str, dst_node_str)]
                    # print(src_node_str, rel, dst_node_str)
                    edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[3])
            with open(os.path.join(stream_data_path, "benign", "stream", "stream-wget-{}.txt".format(gid)), "r") as f:
                for line in f:
                    rel = line.strip().split(" ")[2].split(":")[2]
                    src_node_str = line.strip().split(" ")[0]
                    dst_node_str = line.strip().split(" ")[1]
                    if rel in schemas:
                        schemas[rel].append((src_node_str, dst_node_str))
                    else:
                        schemas[rel] = [(src_node_str, dst_node_str)]
                    edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[5])
            graph_data = {}
            time_feature_rel_dict = dict()
            for k, v in schemas.items():
                rel = k
                relations.add(rel)

                edge_schema = (dgl.NTYPE, rel, dgl.NTYPE)
                src_idx, dst_idx = [], []
                edge_timestamp_list = []
                for src_node, dst_node in v:
                    src_idx.append(int(src_node))
                    dst_idx.append(int(dst_node))
                for i in range(len(src_idx)):
                    m, s = divmod(edge_timestamp[(src_idx[i], rel, dst_idx[i])], 60)
                    h, m = divmod(m, 60)
                    h_m_s = [[h], [m], [s]]
                    edge_timestamp_list.append(h_m_s)
                
                graph_data[edge_schema] = (src_idx, dst_idx)
                time_feature = np.array(edge_timestamp_list)
                print(torch.from_numpy(time_feature))
                time_feature_rel_dict[edge_schema] = torch.from_numpy(time_feature)
            # print(graph_data)
            tmp_g = dgl.heterograph(graph_data)
            tmp_g.edata['timestamp'] = time_feature_rel_dict
            tmp_g = dgl.to_simple(tmp_g, copy_edata=True)

            graphs.append(tmp_g)
            labels.append(label)
        if args.attack == 'sc1' or args.attack == 'both':
            label = 1
            for gid in range(25):
                schemas = dict()
                edge_timestamp = dict()
                with open(os.path.join(stream_data_path, "attack_baseline", "base", "base-wget-attack-baseline-{}.txt".format(gid)), "r") as f:
                    for line in f:
                        rel = line.strip().split(" ")[2].split(":")[2]
                        src_node_str = line.strip().split(" ")[0]
                        dst_node_str = line.strip().split(" ")[1]
                        if rel in schemas:
                            schemas[rel].append((src_node_str, dst_node_str))
                        else:
                            schemas[rel] = [(src_node_str, dst_node_str)]
                        # print(src_node_str, rel, dst_node_str)
                        edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[3])
                with open(os.path.join(stream_data_path, "attack_baseline", "stream", "stream-wget-attack-baseline-{}.txt".format(gid)), "r") as f:
                    for line in f:
                        rel = line.strip().split(" ")[2].split(":")[2]
                        src_node_str = line.strip().split(" ")[0]
                        dst_node_str = line.strip().split(" ")[1]
                        if rel in schemas:
                            schemas[rel].append((src_node_str, dst_node_str))
                        else:
                            schemas[rel] = [(src_node_str, dst_node_str)]
                        edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[5])            
                graph_data = {}
                time_feature_rel_dict = dict()
                for k, v in schemas.items():
                    rel = k
                    relations.add(rel)

                    edge_schema = (dgl.NTYPE, rel, dgl.NTYPE)
                    src_idx, dst_idx = [], []
                    edge_timestamp_list = []
                    for src_node, dst_node in v:
                        src_idx.append(int(src_node))
                        dst_idx.append(int(dst_node))
                    for i in range(len(src_idx)):
                        m, s = divmod(edge_timestamp[(src_idx[i], rel, dst_idx[i])], 60)
                        h, m = divmod(m, 60)
                        h_m_s = [[h], [m], [s]]
                        edge_timestamp_list.append(h_m_s)
                
                    graph_data[edge_schema] = (src_idx, dst_idx)
                    time_feature = np.array(edge_timestamp_list)
                    print(torch.from_numpy(time_feature))
                    time_feature_rel_dict[edge_schema] = torch.from_numpy(time_feature)
                # print(graph_data)
                tmp_g = dgl.heterograph(graph_data)
                tmp_g.edata['timestamp'] = time_feature_rel_dict
                tmp_g = dgl.to_simple(tmp_g, copy_edata=True)

                graphs.append(tmp_g)
                labels.append(label)
        if args.attack == 'sc2' or args.attack == 'both':
            label = 1
            for gid in range(25):
                schemas = dict()
                edge_timestamp = dict()
                with open(os.path.join(stream_data_path, "attack_interval", "base", "base-wget-attack-interval-{}.txt".format(gid)), "r") as f:
                    for line in f:
                        rel = line.strip().split(" ")[2].split(":")[2]
                        src_node_str = line.strip().split(" ")[0]
                        dst_node_str = line.strip().split(" ")[1]
                        if rel in schemas:
                            schemas[rel].append((src_node_str, dst_node_str))
                        else:
                            schemas[rel] = [(src_node_str, dst_node_str)]
                        # print(src_node_str, rel, dst_node_str)
                        edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[3])
                with open(os.path.join(stream_data_path, "attack_interval", "stream", "stream-wget-attack-interval-{}.txt".format(gid)), "r") as f:
                    for line in f:
                        rel = line.strip().split(" ")[2].split(":")[2]
                        src_node_str = line.strip().split(" ")[0]
                        dst_node_str = line.strip().split(" ")[1]
                        if rel in schemas:
                            schemas[rel].append((src_node_str, dst_node_str))
                        else:
                            schemas[rel] = [(src_node_str, dst_node_str)]
                        edge_timestamp[(int(src_node_str),rel, int(dst_node_str))] = int(line.strip().split(" ")[2].split(":")[5])            
                graph_data = {}
                time_feature_rel_dict = dict()
                for k, v in schemas.items():
                    rel = k
                    relations.add(rel)

                    edge_schema = (dgl.NTYPE, rel, dgl.NTYPE)
                    src_idx, dst_idx = [], []
                    edge_timestamp_list = []
                    for src_node, dst_node in v:
                        src_idx.append(int(src_node))
                        dst_idx.append(int(dst_node))
                    for i in range(len(src_idx)):
                        m, s = divmod(edge_timestamp[(src_idx[i], rel, dst_idx[i])], 60)
                        h, m = divmod(m, 60)
                        h_m_s = [[h], [m], [s]]
                        edge_timestamp_list.append(h_m_s)
                
                    graph_data[edge_schema] = (src_idx, dst_idx)
                    time_feature = np.array(edge_timestamp_list)
                    print(torch.from_numpy(time_feature))
                    time_feature_rel_dict[edge_schema] = torch.from_numpy(time_feature)
                # print(graph_data)
                tmp_g = dgl.heterograph(graph_data)
                tmp_g.edata['timestamp'] = time_feature_rel_dict
                tmp_g = dgl.to_simple(tmp_g, copy_edata=True)

                graphs.append(tmp_g)
                labels.append(label)    
    
    relations = list(relations)
    relations.sort()
    for i in range(int(len(labels)/5)):
        cv_fold = [0,1,2,3,4]
        random.shuffle(cv_fold)
        cv_folds.extend(cv_fold)
    
    data_path = stream_data_path
    with open(f'{data_path}/{dataset}_relations.txt', 'w') as f:
        f.write(
            '\n'.join(relations)
        )

    for graph in graphs:
        graph.ndata['feat'] = torch.stack([
            graph.in_degrees(etype=rel) if rel in graph.etypes
            else torch.zeros_like(graph.nodes())
            for rel in relations
        ], dim=1)

    dgl.save_graphs(
        f'{data_path}/{dataset}_dgl_graphs.bin',
        graphs,
        {'label': torch.IntTensor(labels),
         'cv_fold': torch.IntTensor(cv_folds)}
    )
    dgl.save_graphs(
        f'{data_path}/{dataset}_dgl_graphs_homo.bin',
        [dgl.to_simple(dgl.to_homogeneous(g, ndata=['feat']), copy_ndata=True) for g in graphs],
        {'label': torch.IntTensor(labels),
         'cv_fold': torch.IntTensor(cv_folds)}
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help='Name of the dataset. Options are "wget" or "streamspot".')
    parser.add_argument('attack', type=str, default='both',
                        help='Name of the attack set. Options are "sc1", "sc2", or "both". (for wget dataset only)')
    parser.add_argument('timestamp', type=bool, default=False,
                        help='set timestamp for temporary feature')
    
    args = parser.parse_args()

    main(args)


# try code
'''
    else:
        # time_feature = list()
        time_dict_with_graph = dict()
        for gid in range(0, 600):
            schemas = dict()
            node_dict_with_timestamp = dict()
            edge_timestamp = dict()
            node_timestamp = dict()
            node_set = set()
            graph_name, label = map(gid)
            with open(os.path.join(stream_data_path, graph_name + "_data", "base_train", "base-{}-v2-{}.txt".format(graph_name, gid)), "r") as f:
                for line in f:
                    rel = line.strip().split(" ")[2].split(":")[2]
                    src_node_type = line.strip().split(" ")[2].split(":")[0]
                    dst_node_type = line.strip().split(" ")[2].split(":")[1]
                    src_node_str = line.strip().split(" ")[0]
                    dst_node_str = line.strip().split(" ")[1]
                    node_set.add(src_node_str)
                    node_set.add(dst_node_str)
                    # node_dict_with_timestamp[src_node_str] = int(line.strip().split(" ")[2].split(":")[3])
                    # node_dict_with_timestamp[dst_node_str] = int(line.strip().split(" ")[2].split(":")[3])
                    if rel in schemas:
                        schemas[rel].append((src_node_str, dst_node_str))
                    else:
                        schemas[rel] = [(src_node_str, dst_node_str)]
                    edge_timestamp[(src_node_str,rel, dst_node_str)] = int(line.strip().split(" ")[2].split(":")[3])
                f.close()
            with open(os.path.join(stream_data_path, graph_name + "_data", "stream_train", "stream-{}-v2-{}.txt".format(graph_name, gid)), "r") as f:
                for line in f:   
                    rel = line.strip().split(" ")[2].split(":")[2]
                    src_node_type = line.strip().split(" ")[2].split(":")[0]
                    dst_node_type = line.strip().split(" ")[2].split(":")[1]
                    src_node_str = line.strip().split(" ")[0]
                    dst_node_str = line.strip().split(" ")[1]
                    node_set.add(src_node_str)
                    node_set.add(dst_node_str)
                    # node_dict_with_timestamp[src_node_str] = int(line.strip().split(" ")[2].split(":")[5])
                    # node_dict_with_timestamp[dst_node_str] = int(line.strip().split(" ")[2].split(":")[5])
                    if rel in schemas:
                        schemas[rel].append((src_node_str, dst_node_str))
                    else:
                        schemas[rel] = [(src_node_str, dst_node_str)]
                    edge_timestamp[(src_node_str,rel, dst_node_str)] = int(line.strip().split(" ")[2].split(":")[5])
                f.close()
            # TODO: 时 分 秒组成三维向量 将时间向量组织到边特征上
            # get json time_dict
            # print(schemas)
            # print(edge_timestamp)
            # print(node_dict_with_timestamp)
            graph_data, node_id_map = {}, {}
            src_idx_list = list()
            dst_idx_list = list()
            time_feature_list = list()
            time_feature_rel_dict = dict()
            rel_list = list()
            i = 0
            for k, v in schemas.items():
                # print(k)
                # print(v)
                rel = k
                relations.add(rel)

                edge_schema = (dgl.NTYPE, rel, dgl.NTYPE)
                src_idx, dst_idx = [], []
                # src_timestamp, dst_timestamp = [], []
                edge_timestamp_list = []
                for src_node, dst_node in v:
                    h_m_s = list()
                    if src_node in node_id_map:
                        node = node_id_map[src_node]
                    else:
                        node_id_map[src_node] = node = i
                        i += 1

                    # src_timestamp.append(node_dict_with_timestamp[src_node])

                    src_idx.append(node)
                    # node_timestamp[node] = np.array(node_dict_with_timestamp[src_node])
                    
                    if dst_node in node_id_map:
                        node = node_id_map[dst_node]
                    else:
                        node_id_map[dst_node] = node = i
                        i += 1

                    # dst_timestamp.append(node_dict_with_timestamp[dst_node])

                    dst_idx.append(node)
                    # print(edge_timestamp[src_node, rel, dst_node])
                    m, s = divmod(edge_timestamp[(src_node, rel, dst_node)], 60)

                    h, m = divmod(m, 60)
                    h_m_s = [h, m, s]
                    edge_timestamp_list.append(h_m_s)
                    # node_timestamp[node] = np.array(node_dict_with_timestamp[dst_node])

                graph_data[edge_schema] = (src_idx, dst_idx)
                # print(dst_timestamp)
                # dst_timestamp_np = np.array(dst_timestamp)
                # src_timestamp_np = np.array(src_timestamp)
                # #TODO: 使用时间戳计算差值 赋值为时间特征 仅仅只有一个差值而已
                # time_feature = dst_timestamp_np - src_timestamp_np
                # print(len(time_feature))
                # src_idx_list.append(src_idx)
                # dst_idx_list.append(dst_idx)
                # rel_list.append(rel)
                time_feature = np.array(edge_timestamp_list)
                # print(time_feature)
                time_feature_rel_dict[edge_schema] = torch.from_numpy(time_feature).view(-1, 3)
                # time_feature_list.append(time_feature)
                # print(time_feature)
                

            tmp_g = dgl.heterograph(graph_data)
            # for k, v in node_timestamp.items():
            #     # print(v)
            #     tmp_g.nodes[k].data['timestamp'] = torch.from_numpy(v).view(1,1)
            tmp_g.edata['timestamp'] = time_feature_rel_dict
            # print(tmp_g.edata['timestamp'])
            # for k, v in time_feature_rel_dict.items():
            #     # print(j)
            #     # print(tmp_g.edges(etype=rel_list[j]))
            #     # print(len(time_feature_list[j]))
            #     tmp_g.edges[k].data['timestamp'] = torch.from_numpy(v)
            tmp_g = dgl.to_simple(tmp_g, copy_edata=True) # this func del timestamp
            # print(tmp_g.edata['timestamp'])
            # print(time_feature_rel_dict)
            time_dict_with_graph[tmp_g] = time_feature_rel_dict
            graphs.append(tmp_g)
            labels.append(label)
            cv_folds.append(randint(0, 4))
'''