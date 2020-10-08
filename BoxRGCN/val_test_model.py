import torch
import pickle
from ruud_full_model import ruud_model
import Github_downloads.mpqe_master.mpqe.data_utils as data_utils
from ruud_loss import closest
import sys
import logging
from time import time

device = "cuda"
query_structure_dict = {"1-chain":"1p", "2-chain":"2p", "3-chain":"3p", "2-inter":"2i", "3-inter":"3i", "3-chain_inter":"ip", "3-inter_chain":"pi", "1p":"1p", "2p":"2p", "3p":"3p", "2i":"2i", "3i":"3i", "ip":"ip", "pi":"pi"}

if len(sys.argv) == 1:
    data_set = "AIFB"
    model_mode = "validate"
    data_set_path = "./BoxRGCN_Data/" + data_set + "/"
    model_path = "./out/"
    store_path = "./val_test_out/"
elif len(sys.argv) == 2:
    data_set = sys.argv[1]
    model_mode = "validate"
    data_set_path = "./BoxRGCN_Data/" + data_set + "/"
    model_path = "./out/"
    store_path = "./val_test_out/"
elif len(sys.argv) == 3:
    data_set = sys.argv[1]
    model_mode = sys.argv[2]
    data_set_path = "./BoxRGCN_Data/" + data_set + "/"
    model_path = "./out/"
    store_path = "./val_test_out/"
elif len(sys.argv) == 4:
    data_set = sys.argv[1]
    model_mode = sys.argv[2]
    data_set_path = sys.argv[3]
    model_path = "./out/"
    store_path = "./val_test_out/"
elif len(sys.argv) == 5:
    data_set = sys.argv[1]
    model_mode = sys.argv[2]
    data_set_path = sys.argv[3]
    model_path = sys.argv[4]
    store_path = "./val_test_out/"
else:
    data_set = sys.argv[1]
    model_mode = sys.argv[2]
    data_set_path = sys.argv[3]
    model_path = sys.argv[4]
    store_path = sys.argv[5]

logging.basicConfig(filename=store_path + "model_results.log",level=logging.INFO)

query_data_path = data_set_path + "query_data/" + model_mode + "/"

t_start_load = time()

with open(model_path + "BoxRGCN_params.pkl", "rb") as f:
    model_creation_params = pickle.load(f)

num_nodes = model_creation_params["num_nodes"]
num_relations = model_creation_params["num_relations"]
emb_dim = model_creation_params["emb_dim"]
graph, _, _ = data_utils.load_graph(data_set_path, emb_dim)
readout = model_creation_params["readout"]
scatter_op = model_creation_params["scatter_op"]
num_layers = model_creation_params["num_layers"]
num_bases = model_creation_params["num_bases"]
device = model_creation_params["device"]

my_model = ruud_model(num_nodes=num_nodes, num_relations=num_relations, emb_dim=emb_dim, graph=graph,
                      readout=readout, scatter_op=scatter_op, num_layers=num_layers, num_bases=num_bases,
                      store_device=device)
my_model.load_state_dict(torch.load(model_path + "BoxRGCN_model.pt"))
my_model.load_state_dict(my_model.state_dict())
my_model = my_model.to(device)
my_model.eval()

node_embeddings = torch.load(model_path + "BoxRGCN_embedding.pt")

edge_index = torch.load(data_set_path + "edge_index.pt").to(device)
edge_type = torch.load(data_set_path + "edge_type.pt").to(device)

with open(data_set_path + "type_dict.pkl", "rb") as f:
    type_dict = pickle.load(f)

nodes = set()
for key in type_dict:
    nodes |= type_dict[key]
num_nodes = len(nodes)

all_nodes = set(range(num_nodes))

with open(data_set_path + "type_dict.pkl", "rb") as f:
    type_dict = pickle.load(f)

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

dimensionality = int(emb_dim/2)

query_results = []

total_num_true_positives = 0
total_num_false_positives = 0
total_num_true_negatives = 0
total_num_false_negatives = 0

t_stop_load = time()

if model_mode == "validate":
    print("Starting Validation!   ({:.3f}s)".format(t_stop_load-t_start_load))
elif model_mode == "test":
    print("Starting Test!   ({:.3f}s)".format(t_stop_load - t_start_load))

number_of_queries = len(all_queries)
print_number = round(number_of_queries/100)
print_number = print_number if print_number else 1

query_number = 0
num_targets = 0
total_num_targets = 0
query_list = []

column_titles = "   Percentage Complete|     |      Time Taken|     |Number of Targets|     |Structures Used"

print(column_titles)
logging.info(column_titles)

t_start_val_test = time()
t_start = time()

with torch.no_grad():
    for query_dict in all_queries:
        query = query_dict["query"]
        raw_query = query_dict["raw_query"]
        var_ids = query_dict["var_ids"]
        q_graphs = query_dict["q_graphs"]

        targets = raw_query["targets"]
        num_targets += len(targets)
        total_num_targets += len(targets)

        query_embedding = my_model(edge_index=edge_index, edge_type=edge_type, queries=[query], raw_queries=[raw_query],
                                   embeddings=node_embeddings, var_ids=var_ids, q_graphs=q_graphs)

        query_embedding = query_embedding.squeeze()
        query_center = query_embedding[:dimensionality]
        query_offset = query_embedding[dimensionality:]
        positive_offset = query_center + query_offset
        negative_offset = query_center - query_offset
        query_minimum = torch.min(positive_offset, negative_offset)
        query_maximum = torch.max(positive_offset, negative_offset)

        found_answers = set()
        indexer = 0
        for entity in node_embeddings:
            closest_point = closest(query_embedding, entity)

            is_answer = True
            for i in range(dimensionality):
                if not (torch.ge(closest_point[i], query_minimum[i]) and torch.ge(query_maximum[i], closest_point[i])):
                    is_answer = False

            if is_answer:
                found_answers |= {indexer}

            indexer += 1

        true_answers = query_dict["raw_query"]["targets"]

        true_positives = true_answers & found_answers
        false_positives = found_answers - true_answers
        false_negatives = true_answers - found_answers
        true_negatives = ((all_nodes - true_positives) - false_positives) - false_negatives


        num_true_positives = len(true_positives)
        num_false_positives = len(false_positives)
        num_true_negatives = len(true_negatives)
        num_false_negatives = len(false_negatives)

        total_num_true_positives += num_true_positives
        total_num_false_positives += num_false_positives
        total_num_true_negatives += num_true_negatives
        total_num_false_negatives += num_false_negatives

        query_results.append({"num_true_positives":num_true_positives, "num_false_positives":num_false_positives,
                              "num_true_negatives":num_true_negatives, "num_false_negatives":num_false_negatives})

        query_number += 1
        query_list.append(query_structure_dict[query.formula.query_type])
        if query_number % print_number == 0:
            t_end = time()
            percent = query_number / number_of_queries * 100
            row_info = "   {:18.2f}%|     |{:8.3f} seconds|     |{:17d}|     |{}".format(percent, t_end - t_start,
                                                                                         num_targets, query_list)
            print(row_info[3:])
            logging.info(row_info)
            query_list = []
            num_targets = 0
            t_start = time()

t_final = time()

total_time = t_final - t_start_load
training_time = t_final - t_start_val_test

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
logging.info(num_targets__msg)



print(total_num_true_positives)
print(total_num_false_positives)
print(total_num_true_negatives)
print(total_num_false_negatives)
print()
print(query_results)

logging.info(str(total_num_true_positives))
logging.info(str(total_num_false_positives))
logging.info(str(total_num_true_negatives))
logging.info(str(total_num_false_negatives))
logging.info("")
logging.info(str(query_results))
