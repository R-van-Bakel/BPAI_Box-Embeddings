import Github_downloads.mpqe_master.mpqe.data_utils as data_utils
import Github_downloads.mpqe_master.mpqe.encoders as encoders
import Github_downloads.mpqe_master.mpqe.model as model
from Github_downloads.mpqe_master.mpqe.graph import Query

import torch
import pickle
from ruud_data_utils import Subgraph_Generator, edges_to_dict, edges_to_alt_dict, edge_index_to_dict, return_1hop_neighbours, Subgraph_Reformatter, edges_to_outgoing_relation_dict

query_structure = "2p"

graph, feature_modules, node_maps = data_utils.load_graph("BoxRGCN_Data", 128)
features = graph.features

encoder = encoders.DirectEncoder(features, feature_modules)
my_model = model.RGCNEncoderDecoder(graph=graph, enc=encoder, readout="sum", scatter_op="add", dropout=0, weight_decay=0, num_layers=3, shared_layers=False, adaptive=True)

generate_subgraph = getattr(Subgraph_Generator, "generate_subgraph")
reformat_subgraph = getattr(Subgraph_Reformatter, "reformat_subgraph")

edge_index = torch.load("BoxRGCN_Data/edge_index.pt")
edge_type = torch.load("BoxRGCN_Data/edge_type.pt")
edge_index_dict, edge_type_dict = edges_to_dict(edge_index, edge_type)
edge_dict = edges_to_alt_dict(edge_index, edge_type)
edge_indices_dict = edge_index_to_dict(edge_index, edge_type)
relation_dict = edges_to_outgoing_relation_dict(edge_index, edge_type)
processed_graph_data = {"edge_index_dict":edge_index_dict, "edge_type_dict":edge_type_dict, "edge_dict":edge_dict, "edge_indices_dict":edge_indices_dict, "relation_dict":relation_dict}

with open("BoxRGCN_Data/relation_index.pkl", "rb") as f:
    relation_index = pickle.load(f)

with open("BoxRGCN_Data/query_structure_index.pkl", "rb") as f:
    query_structure_index = pickle.load(f)

loaded_graph_data = {"relation_index":relation_index, "query_structure_index":query_structure_index}


###################################### Generate Query: ######################################


raw_query = generate_subgraph(query_structure, processed_graph_data)
neighbourhood_edge_index, neighbourhood_edge_type, neg_targets = return_1hop_neighbours(raw_query, processed_graph_data)

raw_query["negs"] = neg_targets if neg_targets else None
raw_query["hard_negs"] = None

anchors = raw_query["anchors"]
targets = raw_query["targets"]
query_data, _, _, formula = reformat_subgraph(raw_query, query_structure, loaded_graph_data, processed_graph_data)

target = query_data[0][1][0]

query = Query(query_data[0], query_data[-2], query_data[-1])

loss = my_model.forward(formula, [query], [target])
print(loss)
