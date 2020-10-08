import Github_downloads.mpqe_master.mpqe.data_utils as data_utils
from Github_downloads.mpqe_master.mpqe.graph import Query
import torch
import pickle
from ruud_data_utils import Subgraph_Generator, edges_to_dict, edges_to_alt_dict, edge_index_to_dict, return_1hop_neighbours, Subgraph_Reformatter, edges_to_outgoing_relation_dict, ruud_RGCNQueryDataset, remove_query_graph_edge, Validate_Test_Subgraph_Generator, get_query_structures
import sys
import random

###################################### Parameters: ######################################


emb_dim = 128
old_idea = False


###################################### Load and Process Data: ######################################

if len(sys.argv) == 1:
    generate_amount = 142857  # ~(10^6)/7
    sample_style = "equal"
    model_mode = "train"
    data_set = "AIFB"
    query_structures = {"1p", "2p", "3p", "2i", "3i", "ip", "pi"}
elif len(sys.argv) == 2:
    generate_amount = sys.argv[1]
    sample_style = "equal"
    model_mode = "train"
    data_set = "AIFB"
    query_structures = {"1p", "2p", "3p", "2i", "3i", "ip", "pi"}
elif len(sys.argv) == 3:
    generate_amount = sys.argv[1]
    sample_style = sys.argv[2]
    model_mode = "train"
    data_set = "AIFB"
    query_structures = {"1p", "2p", "3p", "2i", "3i", "ip", "pi"}
elif len(sys.argv) == 4:
    generate_amount = sys.argv[1]
    sample_style = sys.argv[2]
    model_mode = sys.argv[3]
    data_set = "AIFB"
    query_structures = {"1p", "2p", "3p", "2i", "3i", "ip", "pi"}
elif len(sys.argv) == 5:
    generate_amount = sys.argv[1]
    sample_style = sys.argv[2]
    model_mode = sys.argv[3]
    data_set = sys.argv[4]
    query_structures = {"1p", "2p", "3p", "2i", "3i", "ip", "pi"}
else:
    generate_amount = sys.argv[1]
    sample_style = sys.argv[2]
    model_mode = sys.argv[3]
    data_set = sys.argv[4]
    query_structures = set(sys.argv[5].split(","))

try:
    generate_amount = int(sys.argv[1])
except:
    raise TypeError(f"First argument \"{sys.argv[1]}\" was no integer.")

if not (sample_style == "equal" or sample_style == "free"):
    raise ValueError(f"\"sample_style\" should be equal to \"equal\" or \"free\"")

if not (model_mode == "train" or model_mode == "validate" or model_mode == "test"):
    raise ValueError("model_mode should be equal to \"validate\" or \"test\"")

if not (data_set == "AIFB" or data_set == "AM" or data_set == "MUTAG"):
    raise ValueError("data_set should be equal to \"AIFB\" or \"AM\" or \"MUTAG\"")

path = "./BoxRGCN_Data/" + data_set + "/"

print("Query Structures:", query_structures)

if model_mode == "train":
    edge_index = torch.load(path + "downsampled_graph/downsampled_edge_index.pt")
    edge_type = torch.load(path + "downsampled_graph/downsampled_edge_type.pt")
elif model_mode == "validate" or model_mode == "test":
    edge_index = torch.load(path + "edge_index.pt")
    edge_type = torch.load(path + "edge_type.pt")

edge_index_dict, edge_type_dict = edges_to_dict(edge_index, edge_type)
edge_dict = edges_to_alt_dict(edge_index, edge_type)
edge_indices_dict = edge_index_to_dict(edge_index, edge_type)
relation_dict = edges_to_outgoing_relation_dict(edge_index, edge_type)

nodes = set()
for i in range(edge_index.shape[1]):
    nodes |= {edge_index[0][i].item(), edge_index[1][i].item()}

with open(path + "type_dict.pkl", "rb") as f:
    type_dict = pickle.load(f)
for key in type_dict:
    type_dict[key] = type_dict[key] & nodes

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
generate_val_test_subgraph = getattr(Validate_Test_Subgraph_Generator, "generate_subgraph_with_edge")

rel_ids = {}
id_rel = 0
for r1 in graph.relations:
    for r2 in graph.relations[r1]:
        rel = (r1, r2[1], r2[0])
        rel_ids[rel] = id_rel
        id_rel += 1

mode_ids = {}
mode_id = 0
for mode in graph.mode_weights:
    mode_ids[mode] = mode_id
    mode_id += 1

with open(path + "downsampled_graph/removed_indices.pkl", "rb") as f:
    removed_indices = pickle.load(f)

# nodes = set()
# for key in type_dict:
#     nodes |= type_dict[key]
# num_nodes = len(nodes)
# num_relations = len(relation_dict)


###################################### Generate Query Data: ######################################

if sample_style == "equal":  # Not implemented fully for validate and test
    for query_structure in sorted(query_structures):

        current_list = []

        for i in range(generate_amount):
            if model_mode == "train":
                raw_query = generate_subgraph(query_structure, processed_graph_data)
            elif model_mode == "validate" or model_mode == "test":
                raw_query = generate_val_test_subgraph(query_structure, processed_graph_data, removed_indices)

            anchors = raw_query["anchors"]
            targets = raw_query["targets"]

            neighbourhood_edge_index, neighbourhood_edge_type, neg_targets = return_1hop_neighbours(raw_query, processed_graph_data, len(targets))

            raw_query["negs"] = neg_targets if neg_targets else None
            raw_query["hard_negs"] = None

            query_data, _, _, formula = reformat_subgraph(raw_query, query_structure, loaded_graph_data, processed_graph_data)

            query = Query(query_data[0], query_data[-2], query_data[-1])

            query_data = ruud_RGCNQueryDataset.get_query_graph(formula, [query], rel_ids, mode_ids)
            var_ids, q_graphs = query_data

            if model_mode == "train" and old_idea:
                remove_query_graph_edge(q_graphs)

            current_list.append({"neighbourhood_edge_index":neighbourhood_edge_index,
                                 "neighbourhood_edge_type":neighbourhood_edge_type, "neg_targets":neg_targets,
                                 "query":query, "raw_query":raw_query, "var_ids":var_ids, "q_graphs":q_graphs})

        used_structures = set()

        for i in range(len(current_list)):
            structures = get_query_structures(current_list[i])

            old = True
            to_change = False
            while old:
                found = False
                for structure in structures:
                    if structure in used_structures:
                        found = True
                        break

                if found:
                    to_change = True
                    if model_mode == "train":
                        raw_query = generate_subgraph(query_structure, processed_graph_data)
                    elif model_mode == "validate" or model_mode == "test":
                        raw_query = generate_val_test_subgraph(query_structure, processed_graph_data, removed_indices)

                    anchors = raw_query["anchors"]
                    targets = raw_query["targets"]

                    neighbourhood_edge_index, neighbourhood_edge_type, neg_targets = return_1hop_neighbours(raw_query,
                                                                                                            processed_graph_data,
                                                                                                            len(targets))

                    raw_query["negs"] = neg_targets if neg_targets else None
                    raw_query["hard_negs"] = None

                    query_data, _, _, formula = reformat_subgraph(raw_query, query_structure, loaded_graph_data,
                                                                  processed_graph_data)

                    query = Query(query_data[0], query_data[-2], query_data[-1])

                    query_data = ruud_RGCNQueryDataset.get_query_graph(formula, [query], rel_ids, mode_ids)
                    var_ids, q_graphs = query_data

                    if model_mode == "train" and old_idea:
                        remove_query_graph_edge(q_graphs)

                    new_query = {"neighbourhood_edge_index":neighbourhood_edge_index,
                                 "neighbourhood_edge_type":neighbourhood_edge_type, "neg_targets":neg_targets,
                                 "query":query, "raw_query":raw_query, "var_ids":var_ids, "q_graphs":q_graphs}

                    structures = get_query_structures(new_query)
                else:
                    old=False
                    used_structures |= structures

            if to_change:
                current_list[i] = new_query

        with open(path + "query_data/" + model_mode + "/query_data_" + query_structure + ".pkl", "wb") as f:
            pickle.dump(current_list, f)
elif sample_style == "free":
    list_1p = []
    list_2p = []
    list_3p = []
    list_2i = []
    list_3i = []
    list_ip = []
    list_pi = []

    structures_1p = set()
    structures_2p = set()
    structures_3p = set()
    structures_2i = set()
    structures_3i = set()
    structures_ip = set()
    structures_pi = set()

    for i in range(generate_amount):
        old = True
        while old:
            query_structure = random.sample(query_structures, 1)[0]

            if model_mode == "train":
                raw_query = generate_subgraph(query_structure, processed_graph_data)
            elif model_mode == "validate" or model_mode == "test":
                raw_query = generate_val_test_subgraph(query_structure, processed_graph_data, removed_indices)

            anchors = raw_query["anchors"]
            targets = raw_query["targets"]

            neighbourhood_edge_index, neighbourhood_edge_type, neg_targets = return_1hop_neighbours(raw_query,
                                                                                                    processed_graph_data,
                                                                                                    len(targets))

            raw_query["negs"] = neg_targets if neg_targets else None
            raw_query["hard_negs"] = None

            query_data, _, _, formula = reformat_subgraph(raw_query, query_structure, loaded_graph_data,
                                                          processed_graph_data)

            query = Query(query_data[0], query_data[-2], query_data[-1])

            query_data = ruud_RGCNQueryDataset.get_query_graph(formula, [query], rel_ids, mode_ids)
            var_ids, q_graphs = query_data

            query_dict = {"neighbourhood_edge_index":neighbourhood_edge_index,
                          "neighbourhood_edge_type":neighbourhood_edge_type, "neg_targets":neg_targets,
                          "query":query, "raw_query":raw_query, "var_ids":var_ids, "q_graphs":q_graphs}

            if model_mode == "train" and old_idea:
                remove_query_graph_edge(q_graphs)

            found = False
            structures = get_query_structures(query_dict)
            sample_structure = random.sample(structures, 1)[0]

            if query_structure == "1p":
                if not sample_structure in structures_1p:
                    old = False
                    structures_1p |= structures
                    list_1p.append(query_dict)
            elif query_structure == "2p":
                if not sample_structure in structures_2p:
                    old = False
                    structures_2p |= structures
                    list_2p.append(query_dict)
            elif query_structure == "3p":
                if not sample_structure in structures_3p:
                    old = False
                    structures_3p |= structures
                    list_3p.append(query_dict)
            elif query_structure == "2i":
                if not sample_structure in structures_2i:
                    old = False
                    structures_2i |= structures
                    list_2i.append(query_dict)
            elif query_structure == "3i":
                if not sample_structure in structures_3i:
                    old = False
                    structures_3i |= structures
                    list_3i.append(query_dict)
            elif query_structure == "ip":
                if not sample_structure in structures_ip:
                    old = False
                    structures_ip |= structures
                    list_ip.append(query_dict)
            else:
                if not sample_structure in structures_pi:
                    old = False
                    structures_pi |= structures
                    list_pi.append(query_dict)

    if model_mode == "train":
        with open(path + "query_data/" + model_mode + "/query_data_1p.pkl", "wb") as f:
            pickle.dump(list_1p, f)

        with open(path + "query_data/" + model_mode + "/query_data_2p.pkl", "wb") as f:
            pickle.dump(list_2p, f)

        with open(path + "query_data/" + model_mode + "/query_data_3p.pkl", "wb") as f:
            pickle.dump(list_3p, f)

        with open(path + "query_data/" + model_mode + "/query_data_2i.pkl", "wb") as f:
            pickle.dump(list_2i, f)

        with open(path + "query_data/" + model_mode + "/query_data_3i.pkl", "wb") as f:
            pickle.dump(list_3i, f)

        with open(path + "query_data/" + model_mode + "/query_data_ip.pkl", "wb") as f:
            pickle.dump(list_ip, f)

        with open(path + "query_data/" + model_mode + "/query_data_pi.pkl", "wb") as f:
            pickle.dump(list_pi, f)
    elif model_mode == "validate" or model_mode == "test":
        validate_amount = round(len(list_1p)/11)
        with open(path + "query_data/validate/query_data_1p.pkl", "wb") as f:
            pickle.dump(list_1p[:validate_amount], f)
        with open(path + "query_data/test/query_data_1p.pkl", "wb") as f:
            pickle.dump(list_1p[validate_amount:], f)

        validate_amount = round(len(list_2p) / 11)
        with open(path + "query_data/validate/query_data_2p.pkl", "wb") as f:
            pickle.dump(list_2p[:validate_amount], f)
        with open(path + "query_data/test/query_data_2p.pkl", "wb") as f:
            pickle.dump(list_2p[validate_amount:], f)

        validate_amount = round(len(list_3p) / 11)
        with open(path + "query_data/validate/query_data_3p.pkl", "wb") as f:
            pickle.dump(list_3p[:validate_amount], f)
        with open(path + "query_data/test/query_data_3p.pkl", "wb") as f:
            pickle.dump(list_3p[validate_amount:], f)

        validate_amount = round(len(list_2i) / 11)
        with open(path + "query_data/validate/query_data_2i.pkl", "wb") as f:
            pickle.dump(list_2i[:validate_amount], f)
        with open(path + "query_data/test/query_data_2i.pkl", "wb") as f:
            pickle.dump(list_2i[validate_amount:], f)

        validate_amount = round(len(list_3i) / 11)
        with open(path + "query_data/validate/query_data_3i.pkl", "wb") as f:
            pickle.dump(list_3i[:validate_amount], f)
        with open(path + "query_data/test/query_data_3i.pkl", "wb") as f:
            pickle.dump(list_3i[validate_amount:], f)

        validate_amount = round(len(list_ip) / 11)
        with open(path + "query_data/validate/query_data_ip.pkl", "wb") as f:
            pickle.dump(list_ip[:validate_amount], f)
        with open(path + "query_data/test/query_data_ip.pkl", "wb") as f:
            pickle.dump(list_ip[validate_amount:], f)

        validate_amount = round(len(list_pi) / 11)
        with open(path + "query_data/validate/query_data_pi.pkl", "wb") as f:
            pickle.dump(list_pi[:validate_amount], f)
        with open(path + "query_data/test/query_data_pi.pkl", "wb") as f:
            pickle.dump(list_pi[validate_amount:], f)
