#!/usr/bin/env python3

from asyncore import read
import os
import logging
from pickletools import read_string1
from re import M
from time import time
from traceback import print_tb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import math
import torch
from torch import nn, optim, set_flush_denormal
import torch.nn.functional as F
from torch.utils.data import DataLoader


import dgl.function as fn
from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch import SumPooling, AvgPooling, MaxPooling, SortPooling
from dgl.utils import expand_as_pair
import numpy as np
from urllib3 import Retry
import dgl
# TODO: Reset parameters methods
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class RelTemporalEncoding(nn.Module):
    '''
        Implement the Temporal Encoding (Sinusoid) function.
    '''
    def __init__(self, n_hid, max_len = 3000, dropout = 0.2):
        super(RelTemporalEncoding, self).__init__()
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term_sin = torch.exp(torch.arange(0, n_hid, 2) *
                             -(math.log(10000.0) / n_hid))
        div_term_cos = torch.exp(torch.arange(0, n_hid-1, 2) *
                             -(math.log(10000.0) / n_hid)) # 奇数特征所以需要修改                    
        emb = nn.Embedding(max_len, n_hid, max_norm=1000.0)
        emb.weight.data[:, 0::2] = torch.sin(position * div_term_sin) / math.sqrt(n_hid) # 奇数个特征的原因所以导致要加一
        emb.weight.data[:, 1::2] = torch.cos(position * div_term_cos) / math.sqrt(n_hid)
        emb.requires_grad = False
        self.emb = emb
        self.lin = nn.Linear(n_hid, n_hid)
    def forward(self, t):
        #TODO 每个向量都要做操作
        #t = torch.tensor(t).to(torch.int64)
        # print("****************")
        # print(t.shape)
        def wt_sum(x):
            h_wt = 3600/3661
            m_wt = 60/3661
            s_wt = 1/3661
            return (h_wt*x[0][0] + m_wt*x[0][1] + s_wt*x[0][2]).view(1, -1)
        

        tmp = t[0,:].reshape(1, -1)
        # print(tmp)
        # A = self.lin(self.emb(tmp)).sum(axis=[0, 1])
        
        A = wt_sum(self.lin(self.emb(tmp)))
        # 时间vector 相加 平均 拓展嵌入
        
        for i in range(1, t.shape[0]):   # 通道数取值
            mp_node_all_src_node = t[i,:].reshape(1, -1)
            A = torch.cat((A, wt_sum(self.lin(self.emb(mp_node_all_src_node)))), 0)
        # A = self.lin(self.emb(t_np[0]))
        # for i in range(1, t.size(0)):
            
        #     A = torch.cat((A, self.lin(self.emb(t_np[i]))), 0)
        # print(A)
        return A.view(-1, 62) # 26 input node feature dim

class GraphClassifier(nn.Module):
    def __init__(
            self,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            agg_type='mean',
            dropout=0.,
            activation=None,
            pool='sum',
            total_latent_dim=None,
            inter_dim=None,
            final_dropout=None,
            sopool_type='bimap',
            timestamp=False
            
    ):
        super(GraphClassifier, self).__init__()
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a

        self.agg_type = agg_type.casefold()
        self.dropout = dropout
        self.activation = activation.casefold()
        self.pool = pool.casefold()
        self.total_latent_dim = total_latent_dim
        self.inter_dim = inter_dim
        self.final_dropout = final_dropout
        self.sopool_type = sopool_type
        self.timestamp = timestamp

        self.embedder = RahmenGraph(
            relations=self.relations,
            feat_dim=self.feat_dim,
            embed_dim=self.embed_dim,
            dim_a=self.dim_a,
            agg_type=self.agg_type,
            dropout=self.dropout,
            activation=self.activation,
            pool=self.pool,
            total_latent_dim=self.total_latent_dim,
            inter_dim=self.inter_dim,
            sopool_type=self.sopool_type,
            timestamp=self.timestamp
        )
        if self.pool == 'sort':
            self.classifier = BinaryClassifierForSort(self.embed_dim)
        elif self.pool == 'sopool':
            if self.sopool_type == 'bimap':
                dense_dim = self.inter_dim * self.inter_dim
            else: # attend
                dense_dim = self.total_latent_dim
            self.classifier = BinaryClassifierForSoPool(dense_dim, final_dropout=self.final_dropout)
        else:
            self.classifier = BinaryClassifier(self.embed_dim)

    def forward(self, graph):
        feat = graph.ndata['feat'].float()  # message passing only supports float dtypes
        time_feat = graph.edata['timestamp']
        # emb_feat = graph.edata['emb_feat']
        # print(time_feat)
        embed = self.embedder(graph, feat, time_feat)
        return self.classifier(embed)

    def train_model(
            self,
            train_dataset,
            batch_size=16,
            EPOCHS=50,
            lr=1e-3,
            weight_decay=0.01,
            accum_steps=1,
            num_workers=2,
            device='cpu',
            model_dir='saved_models/model'
    ):
        self.to(device)

        os.makedirs(model_dir, exist_ok=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )


        start_train = time()
        train_acc, train_p, train_r, train_f1 = [], [], [], []
        train_score = 0
        for epoch in range(EPOCHS):
            self.train()
            self.to(device)
            
            data_iter = tqdm(
                train_loader,
                desc=f'Epoch: {epoch:02}',
                total=len(train_loader),
                position=0
            )

            loss, avg_loss, mean_loss = None, 0., 0.
            for i, (batch_graph, labels) in enumerate(data_iter):
                batch_graph = batch_graph.to(device)
                # print(labels)
                # print(labels.shape)
                labels = labels.squeeze().to(device)
                # print(labels)
                # print(labels.shape)

                # print(batch_graph)
                # print(batch_graph.shape)
                logits = self(batch_graph).squeeze()
                #logits = self(batch_graph)
                # print(logits)
                # print(logits.shape)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                # print(loss)
                loss /= accum_steps  # Normalize for batch accumulation
                loss.backward()

                if ((i+1) % accum_steps == 0) or ((i+1) == len(data_iter)):
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss += loss.item()
                mean_loss = avg_loss / (i+1)
                data_iter.set_postfix({
                    'train_score': train_score,
                    'avg_loss': avg_loss / (i+1)
                })
            logging.info(f'loss {loss:.4f}')
            logging.info(f'mean_loss {mean_loss:.4f}')
            # if epoch % 5 == 0:
            #     acc, p, r, f1, = self.eval_model(
            #         train_dataset, batch_size=batch_size, num_workers=num_workers, device=device
            #     )
            #     train_score = acc

            #     logging.info(f'{epoch:02}: Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} ')

            #     train_acc.append(acc)
            #     train_p.append(p)
            #     train_r.append(r)
            #     train_f1.append(f1)

            #     # Save checkpoint
            #     torch.save({
            #         'epoch': epoch,
            #         'loss': loss,
            #         'model_state_dict': self.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict()
            #     }, f'{model_dir}/checkpoint.pt')
        end_train = time()
        logging.info(f'Total training time... {end_train - start_train:.2f}s')

        # Save final model separately
        model_name = os.path.normpath(model_dir).split(os.sep)[-1]
        

        return train_acc, train_p, train_r, train_f1

    def eval_model(
            self,
            eval_dataset,
            batch_size=16,
            num_workers=2,
            device='cpu'
    ):
        self.eval()
        self.to(device)

        eval_loader = GraphDataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )

        pred_probs, y_true = self.predict(eval_loader)
        #print(y_true)
        num_true = int(sum(y_true))
        #print(num_true)
        sorted_pred = sorted(pred_probs)
        #print(sorted_pred)
        threshold = sorted_pred[-num_true]
        #print(threshold)
        y_pred = [
            1 if pred >= 0.5 else 0
            for pred in pred_probs
        ]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1

    def predict(self, graph_loader, device='cpu'):
        self.eval()
        self.to(device)

        data_iter = tqdm(
            graph_loader,
            desc=f'',
            total=len(graph_loader),
            position=0
        )

        with torch.no_grad():
            preds, labels = [], []
            for batch_graph, batch_labels in data_iter:
                batch_graph = batch_graph.to(device)

                batch_preds = torch.sigmoid(self(batch_graph))

                preds.extend(batch_preds.cpu())
                labels.extend(batch_labels)

        return preds, labels
    def predict_forval(self, graph_loader, device='cpu'):
        self.eval()
        self.to(device)

        data_iter = tqdm(
            graph_loader,
            desc=f'',
            total=len(graph_loader),
            position=0
        )

        with torch.no_grad():
            preds, labels = [], []
            for batch_graph, batch_labels in data_iter:
                batch_graph = batch_graph.to(device)

                # batch_preds = torch.sigmoid(self(batch_graph))

                preds.extend(batch_preds.cpu())
                labels.extend(batch_labels)

        return preds, labels

class RahmenGraph(nn.Module):
    def __init__(
            self,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            agg_type='mean',
            dropout=0.,
            activation=None,
            norm=False,
            pool='sum',
            total_latent_dim=None,
            inter_dim=None,
            sopool_type='bimap',
            timestamp=False
    ):
        super(RahmenGraph, self).__init__()
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a

        self.dropout = dropout
        self.activation = activation
        self.norm = norm
        self.timestamp = timestamp

        self.transform = nn.ModuleDict({
            rel: MessageTransform(
                in_dim=self.feat_dim,
                out_dim=self.embed_dim,
                dropout=self.dropout,
                activation=self.activation,
                norm=self.norm
            )
            for rel in relations
        })
        
        
        self.emb            = RelTemporalEncoding(self.feat_dim)
        self.attention = SemanticAttention(self.num_relations, self.embed_dim, self.dim_a)

        # TODO: Separate node reduce and global readout functions? YES
        self.reduce_fn = self._get_reduce_fn(agg_type)
        self.readout_fn = self._get_readout_fn(pool, total_latent_dim, inter_dim, sopool_type)

    @staticmethod
    def _get_reduce_fn(agg_type):
        if agg_type == 'mean':
            reduce_fn = fn.mean
        
        elif agg_type == 'max':
            reduce_fn = fn.max

        elif agg_type == 'sum':
            reduce_fn = fn.sum

        else:
            raise ValueError('Invalid aggregation function')

        return reduce_fn

    @staticmethod
    def _get_readout_fn(pool, total_latent_dim=None, inter_dim=None, sopool_type='bimap'):
        if pool == 'sum':
            readout_fn = SumPooling()
        elif pool == 'mean':
            readout_fn = AvgPooling()
        elif pool == 'max':
            readout_fn = MaxPooling()
        elif pool == 'sort':
            readout_fn = SortPooling(k=3)

        elif pool == 'sopool':
            readout_fn = SoPooling(total_latent_dim, inter_dim, sopool_type)

        else:
            raise ValueError('Invalid pool function')
        
        return readout_fn

    def forward(self, graph, feat, time_feat):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.embed_dim, device=graph.device)
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # print("11111111111111111111")
            # print(edge_feat)
            # temp_edge_feat = self.emb(edge_feat)
            # print("1111111111")
            # print(temp_edge_feat)
            # print(graph.num_nodes())
            # print(feat_dst.shape)
            # print(feat_src.shape)
            for i, rel in enumerate(self.relations):
                # temp_edge_feat = dict()
                if rel in graph.etypes:

                    # graph.edata['m'] = time_feat
                    
                    # temp_edge_feat[(dgl.NTYPE, rel, dgl.NTYPE)] = self.emb(edge_feat[(dgl.NTYPE, rel, dgl.NTYPE)].sq)
                    graph.srcdata['h'] = feat_src
                    # graph.edata['he'] = {(dgl.NTYPE, rel, dgl.NTYPE) : torch.zeros(graph.num_edges((dgl.NTYPE, rel, dgl.NTYPE)), 26, device=graph.device)}
                    #graph.edata['a'] = temp_edge_feat
                    # TODO: Add node-level attention from GAT
                    # print("**************")
                    # print(graph.edges())
                    # def message_func(edges):
                    #     # print((edges.src['hu']-edges.dst['hv']).squeeze(1))
                    #     # print(edges.dst['hv'])
                    #     # print(edges.src['h'])
                    #     # src, dst, _ = edges.edges()
                    #     sub_time = edges.src['hu'] - edges.dst['hv']
                    #     print(self.emb(sub_time).shape)
                    #     return {'hu':self.emb(sub_time)}
                    # def message_func1(edges):
                    #     print(edges.src['hu'] - edges.src['hu'])
                    #     return {'m' : edges.src['hu'] - edges.dst['hu']}
                    # graph.apply_edges(message_func1, etype=rel)
                    # #print(graph.edata['m'])
                    # graph.srcdata['hu'] = feat_src
                    def message_func(edges):
                        mp = torch.tensor(edges.data['m']).to(torch.int64)
                        time_emb = self.emb(mp)
                        return {'j':edges.src['h'] + time_emb}
                    # print(graph.srcdata['hu'])
                    # def node_udf(nodes):
                    #     # nodes.data['h'] is a tensor of shape (N, 1),
                    #     # nodes.mailbox['m'] is a tensor of shape (N, D, 1),
                    #     # where N is the number of nodes in the batch,
                    #     # D is the number of messages received per node for this node batch
                    #     # print(nodes.data['h'])
                    #     # print(nodes.mailbox['m'].sum(1))
                    #     # sub_time = nodes.data['hu'] - nodes.data['hv']
                    #     print(nodes.mailbox['m'].shape)
                    #     print(nodes.data['hv'].shape)
                    #     print(nodes.data['h'].shape)
                    #     # for i in range(nodes.mailbox['m'].shape[0]):   # 通道数取值
                    #     #     tmp_node_all_src_node = nodes.mailbox['m'][i,:,:]
                    #     #     tmp_node_single_dst_node = node.data['hv'][i,:]
                    #     #         for j in range(nodes.mailbox['m'].shape[1]):
                    #     #             tmp_node_single_src_node = tmp_node_all_src_node[j,:]
                    #     #             self.emb(tmp_node_single_dst_node - tmp_node_single_src_node)
                    #     return {'h': nodes.data['h'] }
                    # def node_udf(nodes):
                    #     print(nodes.mailbox['m'].shape)
                    #     return {'h': nodes.data['h'] }
                    # graph.update_all(fn.v_sub_u('hv', 'hu', 'm'), node_udf, etype=rel)
                    # print(graph.ndata['h'])
                    # TODO he特征接收不到
                    # print(graph.edata['j'])

                    # graph.update_all(
                    #     message_func,
                    #     self.reduce_fn('j', 'neigh'),
                    #     etype=rel
                    # )

                    graph.update_all(
                        fn.copy_u('h', 'm'),
                        self.reduce_fn('m', 'neigh'),
                        etype=rel
                    )
                    h_rel = feat_dst + graph.dstdata['neigh']

                    h[i] = self.transform[rel](h_rel)
                    # if self.timestamp:
                    #     h[i] = self.emb(h[i], )
            # print(h)
            h = self.attention(graph, h)
            # print("after attendtion")
            

        return self.readout_fn(graph, h)


class MessageTransform(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            num_layers=2,
            dropout=0.,
            activation='relu',
            norm=True,
    ):
        super(MessageTransform, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(activation)
        self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True) if norm else None
        if num_layers > 1:
            self.layers = nn.ModuleList([
                nn.Linear(self.in_dim, self.out_dim) if i < num_layers-1
                else nn.Linear(self.out_dim, self.out_dim)
                for i in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Linear(self.in_dim, self.out_dim)
            ])

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = F.relu
        elif activation == 'elu':
            act_fn = F.elu
        elif activation == 'gelu':
            act_fn = F.gelu
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(x)

            if self.norm:
                # TODO: LayerNorm broken. Fix dimensions
                x = self.norm(x)
            if self.activation:
                # print("before activation")
                # print(x)
                x = self.activation(x)
                # print("after activation")
                # print(x)

        return x


class SemanticAttention(nn.Module):
    def __init__(self, num_relations, in_dim, dim_a, out_dim=1, dropout=0.):
        super(SemanticAttention, self).__init__()
        self.num_relations = num_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        self.weights_s1 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.in_dim, self.dim_a)
        )
        self.weights_s2 = nn.Parameter(
            torch.FloatTensor(self.num_relations, self.dim_a, self.out_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights_s1.data, gain=gain)
        nn.init.xavier_uniform_(self.weights_s2.data)

    def forward(self, graph, h, batch_size=64):
        # Shape of input h: (num_relations, num_nodes, dim)
        # Output shape: (num_nodes, dim)
        graph.ndata['h'] = torch.zeros(graph.num_nodes(), h.size(-1), device=graph.device)

        node_loader = DataLoader(
            graph.nodes(),
            batch_size=batch_size,
            shuffle=False,
        )

        for node_batch in node_loader:
            h_batch = h[:, node_batch, :]

            attention = F.softmax(
                torch.matmul(
                    torch.tanh(
                        torch.matmul(h_batch, self.weights_s1)
                    ),
                    self.weights_s2
                ),
                dim=0
            ).squeeze(2)

            attention = self.dropout(attention)

            # TODO: FFT option: https://pytorch.org/docs/stable/generated/torch.fft.rfft2.html
            graph.ndata['h'][node_batch] = torch.einsum('rb,rbd->bd', attention, h_batch)

        return graph.ndata.pop('h')


class BinaryClassifier(nn.Module):
    def __init__(self, embed_dim):
        super(BinaryClassifier, self).__init__()
        self.embed_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, x):
        return self.classifier(x)

class BinaryClassifierForSort(nn.Module):

    def __init__(self, embed_dim):
        super(BinaryClassifierForSort, self).__init__()
        self.embed_dim = 192 #k * feat_size

        self.classifier = nn.Sequential(
            
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, x):
        return self.classifier(x)

class BinaryClassifierForSoPool(nn.Module):
    def __init__(self, dense_dim, final_dropout):
        super(BinaryClassifierForSoPool, self).__init__()
        self.dense_dim = dense_dim
        self.final_dropout = final_dropout
        self.linear1 = nn.Linear(self.dense_dim, 1)

    def forward(self, x):
        # print(x)
        # print(x.shape)
        # print(self.dense_dim)
        score = F.dropout(self.linear1(x), self.final_dropout, training=self.training)
        return score

class SemanticAttention2(nn.Module):
    def __init__(self, relations, in_dim, dim_a, out_dim=1, dropout=0.):
        super(SemanticAttention2, self).__init__()
        self.relations = relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.dropout = nn.Dropout(dropout)

        # TODO: If memory issues, can rewrite to compute one at a time instead
        self.weights_s1 = nn.ModuleDict({
            rel: nn.Parameter(
                torch.FloatTensor(self.in_dim, self.dim_a)
            )
            for rel in self.relations
        })
        self.weights_s2 = nn.ModuleDict({
            rel: nn.Parameter(
                torch.FloatTensor(self.dim_a, self.out_dim)
            )
            for rel in self.relations
        })

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        for rel in self.relations:
            nn.init.xavier_uniform_(self.weights_s1[rel].data, gain=gain)
            nn.init.xavier_uniform_(self.weights_s2[rel].data)

    def forward(self, h, relations):
        # Shape of h: (num_relations, num_nodes, dim)

        weights_s1 = torch.tensor([
            self.weights_s1[rel]
            for rel in relations
        ])
        weights_s2 = torch.tensor([
            self.weights_s2[rel]
            for rel in relations
        ])

        attention = F.softmax(
            torch.matmul(
                torch.tanh(
                    torch.matmul(h, weights_s1)
                ),
                weights_s2
            ),
            dim=0
        ).permute(1, 0, 2)

        attention = self.dropout(attention)

        # Output shape: (num_nodes, num_relations, dim)
        h = torch.matmul(attention, h.permute(1, 0, 2))

        # TODO: Need to squeeze?
        return h

class SoPooling(nn.Module):
    def __init__(self, total_latent_dim, inter_dim, sopool_type='bimap'):
        super(SoPooling, self).__init__()
        self.total_latent_dim = total_latent_dim
        self.inter_dim = inter_dim
        self.sopool_type = sopool_type
        if sopool_type == 'bimap':
            self.dense_dim = self.inter_dim * self.inter_dim
        else:
            self.dense_dim = self.total_latent_dim
         
        self.BiMap = nn.Linear(self.total_latent_dim, self.inter_dim, bias=False)
        self.attend = nn.Linear(self.total_latent_dim, 1)
        self.norm = nn.LayerNorm(self.dense_dim)

    def forward(self, graph, feat):
        # print(feat.shape) # 节点过多导致数据爆炸
        # input: feat num_node * feat_size
        # output: 2dim martix 
        with graph.local_scope():
            if self.sopool_type == 'bimap':
                node_emb = self.BiMap(feat)
                graph_emb = torch.matmul(node_emb.t(), node_emb)         
                ret = graph_emb.view(self.dense_dim)
            if self.sopool_type == 'attend':
                attn_coef = self.attend(feat)
                attn_weights = torch.transpose(attn_coef, 0, 1)
                cur_graph_embeddings = torch.matmul(attn_weights, feat)
                ret = cur_graph_embeddings.view(self.dense_dim)
            return self.norm(ret)
            # return ret