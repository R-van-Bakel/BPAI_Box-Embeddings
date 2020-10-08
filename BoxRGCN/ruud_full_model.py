import torch
from torch.nn import Module
from ruud_RGCN import node_RGCN
from ruud_MPQE import ruud_MPQE

class ruud_model(Module):
    def __init__(self, num_nodes, num_relations, emb_dim, graph, readout='mp', scatter_op='add', num_layers=3, num_bases=None, store_device=None):
        super(ruud_model, self).__init__()

        self.store_device = store_device

        if self.store_device:
            self.RGCN = node_RGCN(num_nodes, num_relations, num_layers, emb_dim, num_bases).to(self.store_device)
            self.MPQE = ruud_MPQE(graph, readout=readout, scatter_op=scatter_op, num_layers=num_layers, num_bases=num_bases).to(self.store_device)
        else:
            self.RGCN = node_RGCN(num_nodes, num_relations, num_layers, emb_dim, num_bases)
            self.MPQE = ruud_MPQE(graph, readout=readout, scatter_op=scatter_op, num_layers=num_layers, num_bases=num_bases)

        # if self.store_device:
        #     self.RGCN.to(self.store_device)
        #     self.MPQE.to(self.store_device)

    def forward(self, edge_index, edge_type, queries, raw_queries, embeddings=None, var_ids=None, q_graphs=None):
        anchor_embeddings = self.RGCN(edge_index, edge_type, embeddings)

        if self.store_device:
            var_ids = var_ids.to(self.store_device)

        raw_query = raw_queries[0]
        anchors = raw_query["anchors"]
        anchor_embeddings_list = []
        for anchor in sorted(anchors):
            anchor_embeddings_list.append(torch.unsqueeze(anchor_embeddings[anchor], 0))

        query_embedding = self.MPQE(anchor_embeddings_list, queries, var_ids, q_graphs)
        return query_embedding
