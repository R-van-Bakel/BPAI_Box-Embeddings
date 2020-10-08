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
gamma = torch.tensor(24., dtype=torch.double, requires_grad=False).to(device)
alpha = torch.tensor(0.2, dtype=torch.double, requires_grad=False).to(device)

# Optimizer
lr=0.0001
weight_decay=0.0005


###################################### Load and Process Data: ######################################



if len(sys.argv) == 1:
    data_set = "AIFB"
else:
    data_set = sys.argv[1]

path = "./BoxRGCN_Data/" + data_set + "/"
query_data_path = path + "query_data/train/"

logging.basicConfig(filename="./out/train_time.log",level=logging.INFO)

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

if len(sys.argv) < 3:
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

    query_structures = {"1p", "2p", "3p", "2i", "3i", "ip", "pi"}
    all_queries = queries_1p + queries_2p + queries_3p + queries_2i + queries_3i + queries_ip + queries_pi
else:
    all_queries = []
    query_structures = set()
    for structure in sys.argv[2].split(","):
        query_structure = query_structure_dict[structure]
        query_structures |= {query_structure}
        with open(query_data_path + "query_data_" + query_structure + ".pkl", "rb") as f:
            queries = pickle.load(f)
        all_queries += queries

if len(sys.argv) < 4:
    num_epochs = 1
else:
    num_epochs = int(sys.argv[3])

random.shuffle(all_queries)
number_of_queries = len(all_queries)
print("Used Query Structures:", query_structures)
print("Number of Queries:", number_of_queries)
print("Number of Epochs:", num_epochs)
print_number = round((number_of_queries*num_epochs)/100)
print_number = print_number if print_number else 1


###################################### Create the Model: ######################################


node_embeddings = create_initial_embeddings(num_nodes, emb_dim, device)
original_node_embeddings = node_embeddings.clone().detach()

my_model = ruud_model(num_nodes=num_nodes, num_relations=num_relations, emb_dim=emb_dim, graph=graph,
                      readout='mp', scatter_op='add', num_layers=num_layers, num_bases=num_bases,
                      store_device=device).to(device)


###################################### Create Optimizer: ######################################


params = list(my_model.parameters())
params.append(node_embeddings)
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
del params
t_stop_load = time()


###################################### Train Model: ######################################


print("Starting Training!   ({:.3f}s)".format(t_stop_load-t_start_load))
query_number = 0
num_targets = 0
total_num_targets = 0
query_list = []

column_titles = "   Percentage Complete|     |      Time Taken|     |Number of Targets|     |Structures Used"
print(column_titles[3:])
logging.info(column_titles)

t_start_train = time()
t_start = time()
my_model.train()

for i in range(num_epochs):
    for query_dict in all_queries:
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        neighbourhood_edge_index = query_dict["neighbourhood_edge_index"].to(device)
        neighbourhood_edge_type = query_dict["neighbourhood_edge_type"].to(device)
        raw_query = query_dict["raw_query"]
        targets = raw_query["targets"]
        neg_targets = query_dict["neg_targets"]
        query = query_dict["query"]
        var_ids = query_dict["var_ids"]
        q_graphs = query_dict["q_graphs"]
        num_targets += len(targets)
        total_num_targets += len(targets)

        ###################################### Run the Model: ######################################


        query_embedding = my_model(edge_index=neighbourhood_edge_index, edge_type=neighbourhood_edge_type,
                                   queries=[query], raw_queries=[raw_query], embeddings=node_embeddings,
                                   var_ids=var_ids, q_graphs=q_graphs)


        ###################################### Calculate Loss: ######################################


        positive_target_embeddings = []
        for target in sorted(targets):
            positive_target_embeddings.append(node_embeddings[target])

        negative_target_embeddings = []
        for neg_target in sorted(neg_targets):
            negative_target_embeddings.append(node_embeddings[neg_target])

        loss = negative_sampling_loss(query_embedding.squeeze(), positive_target_embeddings,
                                      negative_target_embeddings, gamma=gamma, alpha=alpha, device=device)

        del positive_target_embeddings
        del negative_target_embeddings

        loss.backward()
        optimizer.step()

        query_number += 1
        query_list.append(query_structure_dict[query.formula.query_type])
        if query_number % print_number == 0:
            t_end = time()
            percent = (query_number/(number_of_queries * num_epochs)) * 100
            row_info = "   {:18.2f}%|     |{:8.3f} seconds|     |{:17d}|     |{}".format(percent, t_end-t_start, num_targets, query_list)
            print(row_info[3:])
            logging.info(row_info)
            query_list = []
            num_targets = 0
            t_start = time()

t_final = time()

total_time = t_final - t_start_load
training_time = t_final - t_start_train

total_time_msg = f"   Total time:                {total_time}"
average_query_time_msg = f"   Average time (per query):  {training_time/number_of_queries}"
average_target_time_msg = f"   Average time (per target): {training_time/total_num_targets}"
num_queries_msg = f"   Number of queries:         {number_of_queries}"
num_targets__msg = f"   Number of targets:         {total_num_targets}"

print()
print(total_time_msg[3:])
print(average_query_time_msg[3:])
print(average_target_time_msg[3:])
print(num_queries_msg[3:])
print(num_targets__msg[3:])

logging.info("")
logging.info(total_time_msg)
logging.info(average_query_time_msg)
logging.info(average_target_time_msg)
logging.info(num_queries_msg)
logging.info(num_targets__msg + "\n\n\n")


model_creation_params = {"num_nodes":num_nodes, "num_relations":num_relations, "emb_dim":emb_dim, "readout":"mp",
                         "scatter_op":"add", "num_layers":num_layers, "num_bases":num_bases, "device":device}

torch.save(my_model.state_dict(), "./out/BoxRGCN_model.pt")
torch.save(node_embeddings, "./out/BoxRGCN_embedding.pt")
torch.save(original_node_embeddings, "./out/BoxRGCN_original_embedding.pt")
with open("./out/BoxRGCN_params.pkl", "wb") as f:
    pickle.dump(model_creation_params, f)

# g = make_dot(loss)
# g.view()
