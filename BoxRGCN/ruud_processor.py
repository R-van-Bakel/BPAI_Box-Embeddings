import Github_downloads.mpqe_master.mpqe.data_utils as data_utils
import torch
from collections import defaultdict
import pickle
from ruud_data_utils import return_downsampled_graph
import sys
from time import time

def sort_graph(edge_index, edge_type):
    index_result1, index_result2, type_result = zip(*sorted(zip(edge_index[0], edge_index[1], edge_type)))
    index_result1 = list(index_result1)
    index_result2 = list(index_result2)
    type_result = list(type_result)
    return index_result1, index_result2, type_result

t_start = time()
if len(sys.argv) == 1:
    data_set = "AIFB"
else:
    data_set = sys.argv[1]

load_path = "./Github_downloads/mpqe_master/" + data_set +"/processed/"
store_path = "./BoxRGCN_Data/" + data_set +"/"

graph, feature_modules, node_maps = data_utils.load_graph(load_path, 128)

# There are 2061 nodes in AIFB going from 0 to 2060
if data_set == "AIFB":
    type_dict = {"class":set(), "organization":set(), "person":set(), "project":set(), "publication":set(), "topic":set()}
elif data_set == "AM":
    type_dict = {"agent":set(), "aggregation":set(), "physicalthing":set(), "proxy":set(), "webresource":set()}
elif data_set == "MUTAG":
    type_dict = {"atom":set(), "bond":set(), "compound":set(), "structure":set()}
else:
    raise ValueError("The dataset must be \"AIFB\", \"AM\", or \"MUTAG\".")
node_types = {}

print("1/12: Filling type_dict")
for relation in graph.adj_lists:
    from_type = relation[0]
    to_type = relation[2]

    for from_node in graph.adj_lists[relation]:
        type_dict[from_type] |= {from_node}
        node_types[from_node] = {from_type}
        for to_node in graph.adj_lists[relation][from_node]:
            type_dict[to_type] |= {to_node}
            node_types[to_node] = {to_type}

relations = {}
relation_index = {}

print("2/12: Filling relations")
index = 0
for key in graph.relations:
    for relation in graph.relations[key]:
        relations[(key, relation[1], relation[0])] = index
        relations[(relation[0], "reverse: " + relation[1], key)] = index+1
        relation_index[index] = (key, relation[1], relation[0])
        relation_index[index+1] = (relation[0], "reverse: " + relation[1], key)
        index += 2

# print(graph.adj_lists)    #Shape:    {relation: {from: {to,...},...}...}

# some_dict = {}
# for key in graph.adj_lists:
#     some_dict[key] = set()
#     for item in graph.adj_lists[key]:
#         some_dict[key] |= {item}

original_edge_index = [[],[]]
original_edge_type = []

print("3/12: Filling original_edge_index and original_edge_type")
for relation in graph.adj_lists:
    for from_node in graph.adj_lists[relation]:
        for to_node in graph.adj_lists[relation][from_node]:
            original_edge_index[0].append(from_node)
            original_edge_index[1].append(to_node)
            original_edge_type.append(relations[relation])

original_edge_index[0], original_edge_index[1], original_edge_type = sort_graph(original_edge_index, original_edge_type)

reverse_edge_index = [[],[]]
reverse_edge_type = []

print("4/12: Filling reverse_edge_index and reverse_edge_type")
for i in range(len(original_edge_index[0])):
    reverse_edge_index[0].append(original_edge_index[1][i])
    reverse_edge_index[1].append(original_edge_index[0][i])
    reverse_edge_type.append(original_edge_type[i] + 1)

reverse_edge_index[0], reverse_edge_index[1], reverse_edge_type = sort_graph(reverse_edge_index, reverse_edge_type)

print("5/12: Creating edge_index and edge_type")
edge_index = [original_edge_index[0] + reverse_edge_index[0], original_edge_index[1] + reverse_edge_index[1]]
edge_type = original_edge_type + reverse_edge_type

edge_index[0], edge_index[1], edge_type = sort_graph(edge_index, edge_type)

print("6/12: Making tensors")
original_edge_index_tensor = torch.tensor(original_edge_index, dtype=torch.long, requires_grad=False)
original_edge_type_tensor = torch.tensor(original_edge_type, dtype=torch.long, requires_grad=False)
reverse_edge_index_tensor = torch.tensor(reverse_edge_index, dtype=torch.long, requires_grad=False)
reverse_edge_type_tensor = torch.tensor(reverse_edge_type, dtype=torch.long, requires_grad=False)
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, requires_grad=False)
edge_type_tensor = torch.tensor(edge_type, dtype=torch.long, requires_grad=False)

print("7/12: Creating graph_data0 and graph_data1")
graph_data0 = {}
for relation in relations:
    if relation[0] in graph_data0:
        graph_data0[relation[0]].append((relation[2], relation[1]))
    else:
        graph_data0[relation[0]] = [(relation[2], relation[1])]

graph_data1 = {}
for relation in relations:
    graph_data1[relation] = defaultdict(set)

for i in range(len(edge_index[0])):
    if edge_index[0][i] in graph_data1[relation_index[edge_type[i]]]:
        graph_data1[relation_index[edge_type[i]]][edge_index[0][i]] |= {edge_index[1][i]}
    else:
        graph_data1[relation_index[edge_type[i]]][edge_index[0][i]] = {edge_index[1][i]}

with open(load_path + "graph_data.pkl", "rb") as f:
    graph_data = pickle.load(f)

print("8/12: Creating graph_data2")
graph_data2 = graph_data[2]

query_structures = {"1-chain":"1p", "2-chain":"2p", "3-chain":"3p", "2-inter":"2i", "3-inter":"3i", "3-chain_inter":"ip", "3-inter_chain":"pi"}
query_structure_index = {"1p":"1-chain", "2p":"2-chain", "3p":"3-chain", "2i":"2-inter", "3i":"3-inter", "ip":"3-chain_inter", "pi":"3-inter_chain"}

print("9/12: Start Downsampling")
downsampled_original_edge_index_tensor, downsampled_original_edge_type_tensor, removed_original_edge_index_tensor, removed_original_edge_type_tensor = return_downsampled_graph(original_edge_index_tensor, original_edge_type_tensor)

print("10/12: Start creating downsampled objects")
downsampled_original_edge_index = downsampled_original_edge_index_tensor.tolist()
downsampled_original_edge_type = downsampled_original_edge_type_tensor.tolist()

downsampled_reverse_edge_index = [[],[]]
downsampled_reverse_edge_type = []

for i in range(len(downsampled_original_edge_index[0])):
    downsampled_reverse_edge_index[0].append(downsampled_original_edge_index[1][i])
    downsampled_reverse_edge_index[1].append(downsampled_original_edge_index[0][i])
    downsampled_reverse_edge_type.append(downsampled_original_edge_type[i] + 1)

downsampled_reverse_edge_index[0], downsampled_reverse_edge_index[1], downsampled_reverse_edge_type = sort_graph(downsampled_reverse_edge_index, downsampled_reverse_edge_type)

downsampled_edge_index = [downsampled_original_edge_index[0] + downsampled_reverse_edge_index[0], downsampled_original_edge_index[1] + downsampled_reverse_edge_index[1]]
downsampled_edge_type = downsampled_original_edge_type + downsampled_reverse_edge_type

downsampled_edge_index[0], downsampled_edge_index[1], downsampled_edge_type = sort_graph(downsampled_edge_index, downsampled_edge_type)

downsampled_edge_index_tensor = torch.tensor(downsampled_edge_index, dtype=torch.long, requires_grad=False)
downsampled_edge_type_tensor = torch.tensor(downsampled_edge_type, dtype=torch.long, requires_grad=False)



removed_original_edge_index = removed_original_edge_index_tensor.tolist()
removed_original_edge_type = removed_original_edge_type_tensor.tolist()

print("11/12: Start creating removed objects")
removed_reverse_edge_index = [[],[]]
removed_reverse_edge_type = []

for i in range(len(removed_original_edge_index[0])):
    removed_reverse_edge_index[0].append(removed_original_edge_index[1][i])
    removed_reverse_edge_index[1].append(removed_original_edge_index[0][i])
    removed_reverse_edge_type.append(removed_original_edge_type[i] + 1)

removed_reverse_edge_index[0], removed_reverse_edge_index[1], removed_reverse_edge_type = sort_graph(removed_reverse_edge_index, removed_reverse_edge_type)

removed_edge_index = [removed_original_edge_index[0] + removed_reverse_edge_index[0], removed_original_edge_index[1] + removed_reverse_edge_index[1]]
removed_edge_type = removed_original_edge_type + removed_reverse_edge_type

removed_edge_index[0], removed_edge_index[1], removed_edge_type = sort_graph(removed_edge_index, removed_edge_type)

removed_edge_index_tensor = torch.tensor(removed_edge_index, dtype=torch.long, requires_grad=False)
removed_edge_type_tensor = torch.tensor(removed_edge_type, dtype=torch.long, requires_grad=False)



edge_dict = {}
removed_edge_dict = {}

for i in range(edge_index_tensor.shape[1]):
    edge_dict[(edge_index_tensor[0][i].item(), edge_index_tensor[1][i].item(), edge_type_tensor[i].item())] = i

for i in range(removed_edge_index_tensor.shape[1]):
    removed_edge_dict[(removed_edge_index_tensor[0][i].item(), removed_edge_index_tensor[1][i].item(), removed_edge_type_tensor[i].item())] = i

downsampled_indices = []
removed_indices = []

indexer = 0
for edge in edge_dict:
    if edge in removed_edge_dict:
        removed_indices.append(indexer)
    else:
        downsampled_indices.append(indexer)
    indexer += 1

print("12/12: Start saving")
torch.save(original_edge_index_tensor, store_path + "original_edge_index.pt")
torch.save(original_edge_type_tensor, store_path + "original_edge_type.pt")
torch.save(reverse_edge_index_tensor, store_path + "reverse_edge_index.pt")
torch.save(reverse_edge_type_tensor, store_path + "reverse_edge_type.pt")
torch.save(edge_index_tensor, store_path + "edge_index.pt")
torch.save(edge_type_tensor, store_path + "edge_type.pt")
pickle.dump(relations, open(store_path + "relations.pkl", "wb"))
pickle.dump(relation_index, open(store_path + "relation_index.pkl", "wb"))
pickle.dump(type_dict, open(store_path + "type_dict.pkl", "wb"))
pickle.dump(node_types, open(store_path + "node_types.pkl", "wb"))
pickle.dump((graph_data0, graph_data1, graph_data2), open(store_path + "graph_data.pkl", "wb"))
pickle.dump(query_structures, open(store_path + "query_structures.pkl", "wb"))
pickle.dump(query_structure_index, open(store_path + "query_structure_index.pkl", "wb"))
pickle.dump(downsampled_indices, open(store_path + "downsampled_graph/downsampled_indices.pkl", "wb"))
pickle.dump(removed_indices, open(store_path + "downsampled_graph/removed_indices.pkl", "wb"))
torch.save(downsampled_edge_index_tensor, store_path + "downsampled_graph/downsampled_edge_index.pt")
torch.save(downsampled_edge_type_tensor, store_path + "downsampled_graph/downsampled_edge_type.pt")
torch.save(removed_edge_index_tensor, store_path + "downsampled_graph/removed_edge_index.pt")
torch.save(removed_edge_type_tensor, store_path + "downsampled_graph/removed_edge_type.pt")
t_end = time()
print(t_end-t_start)
