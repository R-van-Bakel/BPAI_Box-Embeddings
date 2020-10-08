import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class node_RGCN(torch.nn.Module):
    def __init__(self, num_nodes, num_relations, num_layers, emb_dim, num_bases=None):
        super(node_RGCN, self).__init__()

        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.num_bases = num_bases if num_bases else num_relations

        self.layers = nn.ModuleList()
        rgcn = RGCNConv(in_channels=self.emb_dim,
                        out_channels=self.emb_dim,
                        num_relations=self.num_relations,
                        num_bases=self.num_bases)
        self.layers.append(rgcn)

        for i in range(self.num_layers-1):
            rgcn = RGCNConv(in_channels=self.emb_dim,
                            out_channels=self.emb_dim,
                            num_relations=self.num_relations,
                            num_bases=self.num_bases)

            self.layers.append(rgcn)

    def forward(self, edge_index, edge_type, embeddings=None):

        x = self.layers[0](embeddings, edge_index, edge_type)

        for i in range(1, self.num_layers - 1):
            x = self.layers[i](F.relu(x), edge_index, edge_type)

        return x
