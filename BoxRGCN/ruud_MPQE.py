import torch
import torch.nn as nn
import torch.nn.functional as F
from ruud_data_utils import ruud_RGCNQueryDataset
import random

from Github_downloads.mpqe_master.mpqe.data_utils import RGCNQueryDataset
from Github_downloads.mpqe_master.mpqe.model import RGCNConv, scatter_add, scatter_max, scatter_mean, MLPReadout, TargetMLPReadout

class ruud_MPQE(nn.Module):
    def __init__(self, graph, readout='mp', scatter_op='add', num_layers=3, num_bases=None):
        super(ruud_MPQE, self).__init__()

        self.graph = graph
        self.emb_dim = graph.feature_dims[next(iter(graph.feature_dims))]
        self.mode_embeddings = nn.Embedding(len(graph.mode_weights),
                                            self.emb_dim)
        self.num_layers = num_layers

        num_relations = 0
        for key in graph.relations:
            num_relations += len(graph.relations[key])
        self.num_relations = num_relations

        self.num_bases = num_bases if num_bases else num_relations

        self.mode_ids = {}
        mode_id = 0
        for mode in graph.mode_weights:
            self.mode_ids[mode] = mode_id
            mode_id += 1

        self.rel_ids = {}
        id_rel = 0
        for r1 in graph.relations:
            for r2 in graph.relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.rel_ids[rel] = id_rel
                id_rel += 1

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            rgcn = RGCNConv(in_channels=self.emb_dim,
                            out_channels=self.emb_dim,
                            num_relations=self.num_relations,
                            num_bases=self.num_bases)

            self.layers.append(rgcn)

        if scatter_op == 'add':
            scatter_fn = scatter_add
        elif scatter_op == 'max':
            scatter_fn = scatter_max
        elif scatter_op == 'mean':
            scatter_fn = scatter_mean
        else:
            raise ValueError(f'Unknown scatter op {scatter_op}')

        self.readout_str = readout

        if readout == 'sum':
            self.readout = self.sum_readout
        elif readout == 'max':
            self.readout = self.max_readout
        elif readout == 'mlp':
            self.readout = MLPReadout(self.emb_dim, self.emb_dim, scatter_fn)
        elif readout == 'targetmlp':
            self.readout = TargetMLPReadout(self.emb_dim, scatter_fn)
        elif readout == 'concat':
            self.readout = MLPReadout(self.emb_dim * num_layers, self.emb_dim,
                                      scatter_fn)
        elif readout == 'mp':
            self.readout = self.target_message_readout
        else:
            raise ValueError(f'Unknown readout function {readout}')

    def sum_readout(self, embs, batch_idx, **kwargs):
        return scatter_add(embs, batch_idx, dim=0)

    def max_readout(self, embs, batch_idx, **kwargs):
        out, argmax = scatter_max(embs, batch_idx, dim=0)
        return out

    def target_message_readout(self, embs, batch_size, num_nodes, num_anchors,
                               **kwargs):
        device = embs.device

        non_target_idx = torch.ones(num_nodes, dtype=torch.bool)
        non_target_idx[num_anchors] = 0
        non_target_idx.to(device)

        embs = embs.reshape(batch_size, num_nodes, -1)
        targets = embs[:, ~non_target_idx].reshape(batch_size, -1)

        return targets

    def forward(self, anchor_embeddings, queries, var_ids=None, q_graphs=None):
        batch_size = len(queries)
        formula = queries[0].formula
        query_type = formula.query_type

        if var_ids is None or q_graphs is None:
            query_data = ruud_RGCNQueryDataset.get_query_graph(formula, queries,
                                                          self.rel_ids,
                                                          self.mode_ids)
            var_ids, q_graphs = query_data

        if query_type == "1-chain" or query_type == "1p":
            n_nodes = 2
            num_anchors = 1
        elif query_type == "2-chain" or query_type == "2p":
            n_nodes = 3
            num_anchors = 1
        elif query_type == "3-chain" or query_type == "3p":
            n_nodes = 4
            num_anchors = 1
        elif query_type == "2-inter" or query_type == "2i":
            n_nodes = 3
            num_anchors = 2
        elif query_type == "3-inter" or query_type == "3i":
            n_nodes = 4
            num_anchors = 3
        elif query_type == "3-chain_inter" or query_type == "ip":
            n_nodes = 4
            num_anchors = 2
        elif query_type == "3-inter_chain" or query_type == "pi":
            n_nodes = 4
            num_anchors = 2
        else:
            raise ValueError(f'Unknown query type {query_type}')

        device = next(self.parameters()).device
        q_graphs = q_graphs.to(device)

        x = torch.empty(batch_size, n_nodes, self.emb_dim).to(var_ids.device)
        for i in range(len(anchor_embeddings)):
            x[:, i] = anchor_embeddings[i]
        x[:, num_anchors:] = self.mode_embeddings(var_ids)
        x = x.reshape(-1, self.emb_dim)
        q_graphs.x = x

        h1 = q_graphs.x
        h_layers = []
        for i in range(len(self.layers) - 1):
            h1 = self.layers[i](h1, q_graphs.edge_index, q_graphs.edge_type)
            h1 = F.relu(h1)
            if self.readout_str == 'concat':
                h_layers.append(h1)

        h1 = self.layers[-1](h1, q_graphs.edge_index, q_graphs.edge_type)

        if self.readout_str == 'concat':
            h_layers.append(h1)
            h1 = torch.cat(h_layers, dim=1)

        batch_size = len(queries)

        out = self.readout(embs=h1, batch_idx=q_graphs.batch,
                           batch_size=batch_size, num_nodes=n_nodes,
                           num_anchors=num_anchors)
        return out
