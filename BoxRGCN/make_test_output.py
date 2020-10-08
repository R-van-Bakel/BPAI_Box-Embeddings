import Github_downloads.mpqe_master.mpqe.data_utils as data_utils
from ruud_loss import negative_sampling_loss
import torch
import pickle
from ruud_data_utils import Subgraph_Generator, edges_to_dict, edges_to_alt_dict, edge_index_to_dict, return_1hop_neighbours, Subgraph_Reformatter, edges_to_outgoing_relation_dict, create_initial_embeddings
from ruud_full_model import ruud_model
import random
from time import time
import sys
import logging

# from torchviz import make_dot


###################################### Get Device: ######################################


assert torch.cuda.is_available(), "CUDA not availible."  # Make sure device is running on CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###################################### Hyperparameters: ######################################


# Model
emb_dim = 128
num_bases = None
num_layers = 3

# Loss Function
gamma = torch.tensor(0.5, dtype=torch.long, requires_grad=False).to(device)
alpha = torch.tensor(0.2, dtype=torch.long, requires_grad=False).to(device)

# Optimizer
lr=0.01
weight_decay=0.0005


###################################### Load and Process Data: ######################################


data_set = "AIFB"
model_mode = "train"
path = "./BoxRGCN_Data/" + data_set + "/"
query_data_path = path + "query_data/" + model_mode + "/"

logging.basicConfig(filename="./out/" + model_mode + "_time.log",level=logging.DEBUG)

t_start_load = time()

edge_index = torch.load(path + "edge_index.pt")
edge_type = torch.load(path + "edge_type.pt")
edge_index_dict, edge_type_dict = edges_to_dict(edge_index, edge_type)
edge_dict = edges_to_alt_dict(edge_index, edge_type)
edge_indices_dict = edge_index_to_dict(edge_index, edge_type)
relation_dict = edges_to_outgoing_relation_dict(edge_index, edge_type)
with open(path + "type_dict.pkl", "rb") as f:
    type_dict = pickle.load(f)
with open(path + "relation_index.pkl", "rb") as f:
    relation_index = pickle.load(f)
with open(path + "query_structure_index.pkl", "rb") as f:
    query_structure_index = pickle.load(f)

processed_graph_data = {"edge_index_dict":edge_index_dict, "edge_type_dict":edge_type_dict, "edge_dict":edge_dict, "edge_indices_dict":edge_indices_dict, "relation_dict":relation_dict}
loaded_graph_data = {"relation_index":relation_index, "query_structure_index":query_structure_index}

graph, feature_modules, node_maps = data_utils.load_graph(path, emb_dim)
features = graph.features

generate_subgraph = getattr(Subgraph_Generator, "generate_subgraph")
reformat_subgraph = getattr(Subgraph_Reformatter, "reformat_subgraph")

nodes = set()
for key in type_dict:
    nodes |= type_dict[key]
num_nodes = len(nodes)
num_relations = len(relation_dict)

with open(path + "query_structure_index.pkl", "rb") as f:
    query_structure_index = pickle.load(f)

query_structure_dict = {"1-chain":"1p", "2-chain":"2p", "3-chain":"3p", "2-inter":"2i", "3-inter":"3i", "3-chain_inter":"ip", "3-inter_chain":"pi", "1p":"1p", "2p":"2p", "3p":"3p", "2i":"2i", "3i":"3i", "ip":"ip", "pi":"pi"}

if len(sys.argv) == 1:
    print("loading 1p")
    with open(query_data_path + "query_data_1p.pkl", "rb") as f:
        queries_1p = pickle.load(f)

    print("loading 2p")
    with open(query_data_path + "query_data_2p.pkl", "rb") as f:
        queries_2p = pickle.load(f)

    print("loading 3p")
    with open(query_data_path + "query_data_3p.pkl", "rb") as f:
        queries_3p = pickle.load(f)

    print("loading 2i")
    with open(query_data_path + "query_data_2i.pkl", "rb") as f:
        queries_2i = pickle.load(f)

    print("loading 3i")
    with open(query_data_path + "query_data_3i.pkl", "rb") as f:
        queries_3i = pickle.load(f)

    print("loading ip")
    with open(query_data_path + "query_data_ip.pkl", "rb") as f:
        queries_ip = pickle.load(f)

    print("loading pi")
    with open(query_data_path + "query_data_pi.pkl", "rb") as f:
        queries_pi = pickle.load(f)

    all_queries = queries_1p + queries_2p + queries_3p + queries_2i + queries_3i + queries_ip + queries_pi
else:
    all_queries = []
    for structure in sys.argv[1].split(","):
        query_structure = query_structure_dict[structure]
        print("loading " + query_structure)
        with open(query_data_path + "query_data_" + query_structure + ".pkl", "rb") as f:
            queries = pickle.load(f)
        all_queries += queries

random.shuffle(all_queries)
number_of_queries = len(all_queries)
print_number = round(number_of_queries/100)
print_number = print_number if print_number else 1


###################################### Create the Model: ######################################


node_embeddings = create_initial_embeddings(num_nodes, emb_dim, device)

my_model = ruud_model(num_nodes=num_nodes, num_relations=num_relations, emb_dim=emb_dim, graph=graph,
                      readout='mp', scatter_op='add', num_layers=num_layers, num_bases=num_bases,
                      store_device=device).to(device)

###################################### Store Output: ######################################

model_creation_params = {"num_nodes":num_nodes, "num_relations":num_relations, "emb_dim":emb_dim, "readout":"mp",
                         "scatter_op":"add", "num_layers":num_layers, "num_bases":num_bases, "device":device}

torch.save(my_model.state_dict(), "./out/BoxRGCN_model.pt")
torch.save(node_embeddings, "./out/BoxRGCN_embedding.pt")
with open("./out/BoxRGCN_params.pkl", "wb") as f:
    pickle.dump(model_creation_params, f)
