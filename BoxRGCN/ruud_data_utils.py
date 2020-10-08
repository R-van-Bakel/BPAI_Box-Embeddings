import random
from Github_downloads.mpqe_master.mpqe.graph import Formula, _reverse_relation
from Github_downloads.mpqe_master.mpqe.data_utils import QueryDataset
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

def graph_to_set(edge_index, edge_type):
    graph = set()
    for i in range(len(edge_index[0])):
        graph.add("{}-{}-{}".format(edge_index[0][i].item(), edge_index[1][i].item(), edge_type[i].item()))
    return graph


def edges_to_dict(edge_index, edge_type):
    edge_index_dict = {0: {}, 1: {}}
    edge_type_dict = {}

    for i in range(len(edge_index[0])):
        edge_index_dict[0][i] = edge_index[0][i].item()
        edge_index_dict[1][i] = edge_index[1][i].item()
        edge_type_dict[i] = edge_type[i].item()

    return edge_index_dict, edge_type_dict


# Node fair alternative:

def edges_to_alt_dict(edge_index, edge_type):
    edge_dict = {}
    for i in range(len(edge_index[0])):
        if edge_index[0][i].item() in edge_dict:
            if edge_type[i].item() in edge_dict[edge_index[0][i].item()]:
                edge_dict[edge_index[0][i].item()][edge_type[i].item()].add(edge_index[1][i].item())
            else:
                edge_dict[edge_index[0][i].item()][edge_type[i].item()] = {edge_index[1][i].item()}
        else:
            edge_dict[edge_index[0][i].item()] = {edge_type[i].item(): {edge_index[1][i].item()}}

    return edge_dict


def edge_index_to_dict(edge_index, edge_type):
    index_dict = {}
    for i in range(len(edge_index[0])):
        if edge_index[0][i].item() in index_dict:
            index_dict[edge_index[0][i].item()].add(i)
        else:
            index_dict[edge_index[0][i].item()] = {i}

    return index_dict

def edges_to_outgoing_relation_dict(edge_index, edge_type):
    relation_dict = {}
    for i in range(len(edge_index[0])):
        if edge_type[i].item() in relation_dict:
            relation_dict[edge_type[i].item()].add(edge_index[0][i].item())
        else:
            relation_dict[edge_type[i].item()] = {edge_index[0][i].item()}

    return relation_dict

def return_opposite_relation(relation):
    return relation + 1 if relation % 2 == 0 else relation - 1


def return_1hop_neighbours(query_data, processed_graph_data, max_neg_targets=100):
    neighbour_index = set()
    for key in query_data["anchors"]:
        neighbour_index |= processed_graph_data["edge_indices_dict"][query_data["anchors"][key]]

    if len(neighbour_index) > max_neg_targets:
        sampled_index = set(random.sample(neighbour_index, max_neg_targets))
    else:
        sampled_index = neighbour_index.copy()

    edge_index = [[], []]
    edge_type = []
    neg_neighbourhood_targets = set()

    for index in sorted(sampled_index):
        edge_index[0].append(processed_graph_data["edge_index_dict"][0][index])
        edge_index[1].append(processed_graph_data["edge_index_dict"][1][index])
        edge_type.append(processed_graph_data["edge_type_dict"][index])
        if processed_graph_data["edge_index_dict"][1][index] not in query_data["targets"]:
            neg_neighbourhood_targets |= {processed_graph_data["edge_index_dict"][1][index]}

    if len(neg_neighbourhood_targets) > len(query_data["targets"]):
        neg_targets = set(random.sample(neg_neighbourhood_targets, len(query_data["targets"])))
    else:
        neg_targets = neg_neighbourhood_targets.copy()
        if len(neg_neighbourhood_targets) < len(query_data["targets"]):
            for index in sorted(neighbour_index-sampled_index):
                if processed_graph_data["edge_index_dict"][1][index] not in query_data["targets"]:
                    neg_targets |= {processed_graph_data["edge_index_dict"][1][index]}
                if len(neg_targets) == len(query_data["targets"]):
                    break

    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_type, dtype=torch.long), neg_targets


def saved_format_to_ptgeometric_format(query_data):
    pass


class Subgraph_Generator(object):

    @staticmethod
    def generate_subgraph(query_structure, processed_graph_data, first_int=None, trace=None, removed_indices=None):
        method_name = 'generate_' + str(query_structure)
        method = getattr(Subgraph_Generator, method_name, Subgraph_Generator.handle_error)

        if not trace:
            if not removed_indices:
                return method(processed_graph_data, first_int=first_int)
            else:
                return method(processed_graph_data, first_int=first_int, removed_indices=removed_indices)
        else:
            return method(processed_graph_data, first_int=first_int, trace=trace, removed_indices=removed_indices)

    @staticmethod
    def handle_error(processed_graph_data):
        print("Invalid Query Structure")
        return None

    @staticmethod
    def generate_1p(processed_graph_data, first_int=None):
        if not first_int:
            rand_int = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
        else:
            rand_int = first_int

        anchor_node = processed_graph_data["edge_index_dict"][0][rand_int]
        relation = processed_graph_data["edge_type_dict"][rand_int]
        targets = processed_graph_data["edge_dict"][anchor_node][relation]

        return {"anchors": {0: anchor_node}, "relations": {0: relation}, "targets": targets}

    @staticmethod
    def generate_2p(processed_graph_data, first_int=None, trace=None, removed_indices=None):
        if not trace:
            if not first_int:
                rand_int1 = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
            else:
                rand_int1 = first_int

            anchor_node = processed_graph_data["edge_index_dict"][0][rand_int1]
            relation1 = processed_graph_data["edge_type_dict"][rand_int1]

            variable_nodes = processed_graph_data["edge_dict"][anchor_node][relation1]
            canditates = set()
            for key in variable_nodes:
                canditates |= processed_graph_data["edge_indices_dict"][key]
            rand_int2 = random.sample(canditates, 1)[0]
            relation2 = processed_graph_data["edge_type_dict"][rand_int2]
            while relation2 == return_opposite_relation(relation1):
                found = False
                for node in variable_nodes:
                    if relation2 in processed_graph_data["edge_dict"][node]:
                        if anchor_node in processed_graph_data["edge_dict"][node][relation2]:
                            if not (len(processed_graph_data["edge_dict"][node][relation2]) == 1):
                                found = True
                                break
                if found:
                    break
                else:
                    canditates = canditates - {rand_int2}
                    if not (canditates):
                        if not removed_indices:
                            return Subgraph_Generator.generate_2p(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_2p(processed_graph_data, removed_indices)
                    else:
                        rand_int2 = random.sample(canditates, 1)[0]
                        relation2 = processed_graph_data["edge_type_dict"][rand_int2]
        else:
            anchor_node = trace[0][0]
            relation1 = trace[1][0]

            variable_nodes = processed_graph_data["edge_dict"][anchor_node][relation1]
            relation2 = trace[1][1]

        targets = set()
        for node in variable_nodes:
            if relation2 in processed_graph_data["edge_dict"][node]:
                targets |= processed_graph_data["edge_dict"][node][relation2]

        return {"anchors": {0: anchor_node}, "relations": {0: relation1, 1: relation2}, "targets": targets}

    @staticmethod
    def generate_3p(processed_graph_data, first_int=None, trace=None, removed_indices=None):
        if not trace:
            if not first_int:
                rand_int1 = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
            else:
                rand_int1 = first_int

            anchor_node = processed_graph_data["edge_index_dict"][0][rand_int1]
            relation1 = processed_graph_data["edge_type_dict"][rand_int1]

            variable_nodes1 = processed_graph_data["edge_dict"][anchor_node][relation1]
            canditates1 = set()
            for key in variable_nodes1:
                canditates1 |= processed_graph_data["edge_indices_dict"][key]
            rand_int2 = random.sample(canditates1, 1)[0]
            relation2 = processed_graph_data["edge_type_dict"][rand_int2]
            while relation2 == return_opposite_relation(relation1):
                found = False
                for node in variable_nodes1:
                    if relation2 in processed_graph_data["edge_dict"][node]:
                        if anchor_node in processed_graph_data["edge_dict"][node][relation2]:
                            if not (len(processed_graph_data["edge_dict"][node][relation2]) == 1):
                                found = True
                                break
                if found:
                    break
                else:
                    canditates1 = canditates1 - {rand_int2}
                    if not (canditates1):
                        if not removed_indices:
                            return Subgraph_Generator.generate_3p(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_3p(processed_graph_data, removed_indices)
                    else:
                        rand_int2 = random.sample(canditates1, 1)[0]
                        relation2 = processed_graph_data["edge_type_dict"][rand_int2]
        else:
            anchor_node = trace[0][0]
            relation1 = trace[1][0]

            variable_nodes1 = processed_graph_data["edge_dict"][anchor_node][relation1]
            relation2 = trace[1][1]

        variable_nodes2 = set()
        for node in variable_nodes1:
            if relation2 in processed_graph_data["edge_dict"][node]:
                variable_nodes2 |= processed_graph_data["edge_dict"][node][relation2]

        if not trace or len(trace[1]) == 2:
            canditates2 = set()
            for key in variable_nodes2:
                canditates2 |= processed_graph_data["edge_indices_dict"][key]
            rand_int3 = random.sample(canditates2, 1)[0]
            relation3 = processed_graph_data["edge_type_dict"][rand_int3]
            while relation3 == return_opposite_relation(relation2):
                found = False
                for node in variable_nodes2:
                    reverse_relation = processed_graph_data["edge_dict"][node][return_opposite_relation(relation2)]
                    reverse_candidates = variable_nodes2 & reverse_relation
                    if len(reverse_relation) > len(reverse_candidates):
                        found = True
                        break

                if found:
                    break
                else:
                    canditates2 = canditates2 - {rand_int3}
                    if not (canditates2):
                        if not removed_indices:
                            return Subgraph_Generator.generate_3p(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_3p(processed_graph_data, removed_indices)
                    else:
                        rand_int3 = random.sample(canditates2, 1)[0]
                        relation3 = processed_graph_data["edge_type_dict"][rand_int3]
        else:
            relation3 = trace[1][2]

        targets = set()
        for node in variable_nodes2:
            if relation3 in processed_graph_data["edge_dict"][node]:
                targets |= processed_graph_data["edge_dict"][node][relation3]

        return {"anchors": {0: anchor_node}, "relations": {0: relation1, 1: relation2, 2: relation3},
                "targets": targets}

    @staticmethod
    def generate_2i(processed_graph_data, first_int=None, removed_indices=None):
        if not first_int:
            rand_int1 = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
        else:
            rand_int1 = first_int

        anchor_node1 = processed_graph_data["edge_index_dict"][0][rand_int1]
        relation1 = processed_graph_data["edge_type_dict"][rand_int1]

        variable_nodes = processed_graph_data["edge_dict"][anchor_node1][relation1]
        canditates = set()
        for key in variable_nodes:
            canditates |= processed_graph_data["edge_indices_dict"][key]
        rand_int2 = random.sample(canditates, 1)[0]
        relation2 = processed_graph_data["edge_type_dict"][rand_int2]
        while relation2 == return_opposite_relation(relation1):
            found = False
            for node in variable_nodes:
                if not (len(processed_graph_data["edge_dict"][node][relation2]) == 1):
                    found = True
                    anchor_node2 = \
                    random.sample(processed_graph_data["edge_dict"][node][relation2] - {anchor_node1}, 1)[0]
                    break
            if found:
                break
            else:
                canditates = canditates - {rand_int2}
                if not (canditates):
                    if not removed_indices:
                        return Subgraph_Generator.generate_2i(processed_graph_data)
                    else:
                        return Validate_Test_Subgraph_Generator.generate_2i(processed_graph_data, removed_indices)
                else:
                    rand_int2 = random.sample(canditates, 1)[0]
                    relation2 = processed_graph_data["edge_type_dict"][rand_int2]

        if relation2 != return_opposite_relation(relation1):
            anchor_node2_candidates = set()
            for node in variable_nodes:
                if relation2 in processed_graph_data["edge_dict"][node]:
                    anchor_node2_candidates |= processed_graph_data["edge_dict"][node][relation2]
            anchor_node2 = random.sample(anchor_node2_candidates, 1)[0]

        relation2 = return_opposite_relation(relation2)
        targets = variable_nodes & processed_graph_data["edge_dict"][anchor_node2][relation2]

        return {"anchors": {0: anchor_node1, 1: anchor_node2}, "relations": {0: relation1, 1: relation2},
                "targets": targets}

    @staticmethod
    def generate_3i(processed_graph_data, first_int=None, removed_indices=None):
        if not first_int:
            rand_int1 = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
        else:
            rand_int1 = first_int

        anchor_node1 = processed_graph_data["edge_index_dict"][0][rand_int1]
        relation1 = processed_graph_data["edge_type_dict"][rand_int1]

        variable_nodes1 = processed_graph_data["edge_dict"][anchor_node1][relation1]
        canditates1 = set()
        for key in variable_nodes1:
            canditates1 |= processed_graph_data["edge_indices_dict"][key]
        rand_int2 = random.sample(canditates1, 1)[0]
        relation2 = processed_graph_data["edge_type_dict"][rand_int2]
        while relation2 == return_opposite_relation(relation1):
            found = False
            for node in variable_nodes1:
                if not (len(processed_graph_data["edge_dict"][node][relation2]) == 1):
                    found = True
                    anchor_node2 = \
                    random.sample(processed_graph_data["edge_dict"][node][relation2] - {anchor_node1}, 1)[0]
                    break
            if found:
                break
            else:
                canditates1 = canditates1 - {rand_int2}
                if not (canditates1):
                    if not removed_indices:
                        return Subgraph_Generator.generate_3i(processed_graph_data)
                    else:
                        return Validate_Test_Subgraph_Generator.generate_3i(processed_graph_data, removed_indices)
                else:
                    rand_int2 = random.sample(canditates1, 1)[0]
                    relation2 = processed_graph_data["edge_type_dict"][rand_int2]

        if relation2 != return_opposite_relation(relation1):
            anchor_node2_candidates = set()
            for node in variable_nodes1:
                if relation2 in processed_graph_data["edge_dict"][node]:
                    anchor_node2_candidates |= processed_graph_data["edge_dict"][node][relation2]
            anchor_node2 = random.sample(anchor_node2_candidates, 1)[0]

        relation2 = return_opposite_relation(relation2)

        variable_nodes2 = variable_nodes1 & processed_graph_data["edge_dict"][anchor_node2][relation2]
        canditates2 = set()
        for key in variable_nodes2:
            canditates2 |= processed_graph_data["edge_indices_dict"][key]
        rand_int3 = random.sample(canditates2, 1)[0]
        relation3 = processed_graph_data["edge_type_dict"][rand_int3]
        while relation3 == return_opposite_relation(relation1) or relation3 == return_opposite_relation(relation2):
            found = False
            if return_opposite_relation(relation1) == return_opposite_relation(relation2):
                temp_anchors = {anchor_node1, anchor_node2}
            temp_anchors = {anchor_node1} if relation3 == return_opposite_relation(relation1) else {anchor_node2}
            for node in variable_nodes2:
                if processed_graph_data["edge_dict"][node][relation3] - {anchor_node1, anchor_node2}:
                    found = True
                    anchor_node3 = random.sample(processed_graph_data["edge_dict"][node][relation3] - temp_anchors, 1)[
                        0]
                    break
            if found:
                break
            else:
                canditates2 = canditates2 - {rand_int3}
                if not (canditates2):
                    if not removed_indices:
                        return Subgraph_Generator.generate_3i(processed_graph_data)
                    else:
                        return Validate_Test_Subgraph_Generator.generate_3i(processed_graph_data, removed_indices)
                else:
                    rand_int3 = random.sample(canditates2, 1)[0]
                    relation3 = processed_graph_data["edge_type_dict"][rand_int3]

        if relation3 != return_opposite_relation(relation1) and relation3 != return_opposite_relation(relation2):
            anchor_node3_candidates = set()
            for node in variable_nodes2:
                if relation3 in processed_graph_data["edge_dict"][node]:
                    anchor_node3_candidates |= processed_graph_data["edge_dict"][node][relation3]
            anchor_node3 = random.sample(anchor_node3_candidates, 1)[0]

        relation3 = return_opposite_relation(relation3)

        targets = variable_nodes2 & processed_graph_data["edge_dict"][anchor_node3][relation3]

        return {"anchors": {0: anchor_node1, 1: anchor_node2, 2: anchor_node3},
                "relations": {0: relation1, 1: relation2, 2: relation3}, "targets": targets}

    @staticmethod
    def generate_ip(processed_graph_data, first_int=None, trace=None, removed_indices=None):
        if not trace:
            if not first_int:
                rand_int1 = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
            else:
                rand_int1 = first_int

            anchor_node1 = processed_graph_data["edge_index_dict"][0][rand_int1]
            relation1 = processed_graph_data["edge_type_dict"][rand_int1]
            variable_nodes1 = processed_graph_data["edge_dict"][anchor_node1][relation1]

            canditates1 = set()
            for key in variable_nodes1:
                canditates1 |= processed_graph_data["edge_indices_dict"][key]
            rand_int2 = random.sample(canditates1, 1)[0]
            relation2 = processed_graph_data["edge_type_dict"][rand_int2]
            while relation2 == return_opposite_relation(relation1):
                found = False
                for node in variable_nodes1:
                    if not (len(processed_graph_data["edge_dict"][node][relation2]) == 1):
                        found = True
                        anchor_node2 = \
                            random.sample(processed_graph_data["edge_dict"][node][relation2] - {anchor_node1}, 1)[0]
                        break
                if found:
                    break
                else:
                    canditates1 = canditates1 - {rand_int2}
                    if not (canditates1):
                        if not removed_indices:
                            return Subgraph_Generator.generate_ip(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_ip(processed_graph_data, removed_indices)
                    else:
                        rand_int2 = random.sample(canditates1, 1)[0]
                        relation2 = processed_graph_data["edge_type_dict"][rand_int2]

            if relation2 != return_opposite_relation(relation1):
                anchor_node2_candidates = set()
                for node in variable_nodes1:
                    if relation2 in processed_graph_data["edge_dict"][node]:
                        anchor_node2_candidates |= processed_graph_data["edge_dict"][node][relation2]
                anchor_node2 = random.sample(anchor_node2_candidates, 1)[0]

            relation2 = return_opposite_relation(relation2)
            variable_nodes2 = variable_nodes1 & processed_graph_data["edge_dict"][anchor_node2][relation2]

            canditates2 = set()
            for key in variable_nodes2:
                canditates2 |= processed_graph_data["edge_indices_dict"][key]
            rand_int3 = random.sample(canditates2, 1)[0]
            relation3 = processed_graph_data["edge_type_dict"][rand_int3]
            while relation3 == return_opposite_relation(relation1) or relation3 == return_opposite_relation(relation2):
                found = False
                for node in variable_nodes2:
                    if processed_graph_data["edge_dict"][node][relation3] - {anchor_node1, anchor_node2}:
                        found = True
                        break
                if found:
                    break
                else:
                    canditates2 = canditates2 - {rand_int3}
                    if not (canditates2):
                        if not removed_indices:
                            return Subgraph_Generator.generate_ip(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_ip(processed_graph_data, removed_indices)
                    else:
                        rand_int3 = random.sample(canditates2, 1)[0]
                        relation3 = processed_graph_data["edge_type_dict"][rand_int3]
        else:
            anchor_node1 = trace[0][0]
            relation1 = trace[1][0]
            variable_nodes1 = processed_graph_data["edge_dict"][anchor_node1][relation1]
            anchor_node2= trace[0][1]
            relation2 = trace[1][1]
            variable_nodes2 = variable_nodes1 & processed_graph_data["edge_dict"][anchor_node2][relation2]
            relation3 = trace[1][2]


        targets = set()
        for node in variable_nodes2:
            if relation3 in processed_graph_data["edge_dict"][node]:
                targets |= processed_graph_data["edge_dict"][node][relation3]

        return {"anchors": {0: anchor_node1, 1: anchor_node2}, "relations": {0: relation1, 1: relation2, 2: relation3},
                "targets": targets}

    @staticmethod
    def generate_pi(processed_graph_data, first_int=None, trace=None, removed_indices=None):
        if not trace:
            if not first_int:
                rand_int1 = random.randint(0, len(processed_graph_data["edge_index_dict"][0]) - 1)
            else:
                rand_int1 = first_int

            anchor_node1 = processed_graph_data["edge_index_dict"][0][rand_int1]
            relation1 = processed_graph_data["edge_type_dict"][rand_int1]

            variable_nodes1 = processed_graph_data["edge_dict"][anchor_node1][relation1]

            canditates1 = set()
            for key in variable_nodes1:
                canditates1 |= processed_graph_data["edge_indices_dict"][key]
            rand_int2 = random.sample(canditates1, 1)[0]
            relation2 = processed_graph_data["edge_type_dict"][rand_int2]
            while relation2 == return_opposite_relation(relation1):
                found = False
                for node in variable_nodes1:
                    if relation2 in processed_graph_data["edge_dict"][node]:
                        if anchor_node1 in processed_graph_data["edge_dict"][node][relation2]:
                            if not (len(processed_graph_data["edge_dict"][node][relation2]) == 1):
                                found = True
                                break
                if found:
                    break
                else:
                    canditates1 = canditates1 - {rand_int2}
                    if not (canditates1):
                        if not removed_indices:
                            return Subgraph_Generator.generate_pi(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_pi(processed_graph_data, removed_indices)
                    else:
                        rand_int2 = random.sample(canditates1, 1)[0]
                        relation2 = processed_graph_data["edge_type_dict"][rand_int2]
        else:
            anchor_node1 = trace[0][0]
            relation1 = trace[1][0]

            variable_nodes1 = processed_graph_data["edge_dict"][anchor_node1][relation1]
            relation2 = trace[1][1]

        variable_nodes2 = set()
        for node in variable_nodes1:
            if relation2 in processed_graph_data["edge_dict"][node]:
                variable_nodes2 |= processed_graph_data["edge_dict"][node][relation2]

        if not trace or len(trace[1]) == 2:
            canditates2 = set()
            for key in variable_nodes2:
                canditates2 |= processed_graph_data["edge_indices_dict"][key]
            rand_int3 = random.sample(canditates2, 1)[0]
            relation3 = processed_graph_data["edge_type_dict"][rand_int3]
            while relation3 == return_opposite_relation(relation2):
                found = False
                for node in variable_nodes2:
                    reverse_relation = processed_graph_data["edge_dict"][node][return_opposite_relation(relation2)]
                    reverse_candidates = variable_nodes2 & reverse_relation
                    if len(reverse_relation) > len(reverse_candidates):
                        found = True
                        anchor_node2 = \
                        random.sample(processed_graph_data["edge_dict"][node][relation3] - variable_nodes2, 1)[0]
                        break

                if found:
                    break
                else:
                    canditates2 = canditates2 - {rand_int3}
                    if not (canditates2):
                        if not removed_indices:
                            return Subgraph_Generator.generate_pi(processed_graph_data)
                        else:
                            return Validate_Test_Subgraph_Generator.generate_pi(processed_graph_data, removed_indices)
                    else:
                        rand_int3 = random.sample(canditates2, 1)[0]
                        relation3 = processed_graph_data["edge_type_dict"][rand_int3]

            if relation3 != return_opposite_relation(relation2):
                anchor_node2_candidates = set()
                for node in variable_nodes2:
                    if relation3 in processed_graph_data["edge_dict"][node]:
                        anchor_node2_candidates |= processed_graph_data["edge_dict"][node][relation3]
                anchor_node2 = random.sample(anchor_node2_candidates, 1)[0]

            relation3 = return_opposite_relation(relation3)
        else:
            anchor_node2 = trace[0][1]
            relation3 = trace[1][2]

        targets = variable_nodes2 & processed_graph_data["edge_dict"][anchor_node2][relation3]

        return {"anchors": {0: anchor_node1, 1: anchor_node2}, "relations": {0: relation1, 1: relation2, 2: relation3},
                "targets": targets}

class Subgraph_Path_Sampler(object):

    @staticmethod
    def sample_path(raw_format, query_structure, processed_graph_data):
        method_name = 'sample_' + str(query_structure)
        method = getattr(Subgraph_Path_Sampler, method_name, Subgraph_Path_Sampler.handle_error)
        return method(raw_format, processed_graph_data)

    @staticmethod
    def handle_error(raw_format):
        print("Invalid Query Structure")
        return None

    @staticmethod
    def sample_1p(raw_format, processed_graph_data):
        return {0: random.sample(raw_format["targets"], 1)[0]}

    @staticmethod
    def sample_2p(raw_format, processed_graph_data):
        anchor = raw_format["anchors"][0]
        relation1 = raw_format["relations"][0]
        relation2 = raw_format["relations"][1]
        variable = random.sample(processed_graph_data["edge_dict"][anchor][relation1] & processed_graph_data["relation_dict"][relation2], 1)[0]
        target = random.sample(processed_graph_data["edge_dict"][variable][relation2], 1)[0]

        return {0: variable, 1: target}

    @staticmethod
    def sample_3p(raw_format, processed_graph_data):
        anchor = raw_format["anchors"][0]
        relation1 = raw_format["relations"][0]
        relation2 = raw_format["relations"][1]
        relation3 = raw_format["relations"][2]
        variables1 = processed_graph_data["edge_dict"][anchor][relation1] & processed_graph_data["relation_dict"][relation2]
        temp_vars = set()
        for variable in variables1:
            new_vars = processed_graph_data["edge_dict"][variable][relation2] & processed_graph_data["relation_dict"][relation3]
            if new_vars:
                temp_vars |= {variable}

        variables1 = temp_vars.copy()
        variable1 = random.sample(variables1, 1)[0]
        variable2 = random.sample(processed_graph_data["edge_dict"][variable1][relation2] & processed_graph_data["relation_dict"][relation3], 1)[0]

        target = random.sample(processed_graph_data["edge_dict"][variable2][relation3], 1)[0]

        return {0: variable1, 1: variable2, 2: target}

    @staticmethod
    def sample_2i(raw_format, processed_graph_data):
        # anchor1 = raw_format["anchors"][0]
        # anchor2 = raw_format["anchors"][1]
        # relation1 = raw_format["relations"][0]
        # relation2 = raw_format["relations"][1]
        # target = random.choice(processed_graph_data["edge_dict"][anchor1][relation1] & processed_graph_data["edge_dict"][anchor2][relation2])
        #
        # return {0: target}
        return {0: random.sample(raw_format["targets"], 1)[0]}

    @staticmethod
    def sample_3i(raw_format, processed_graph_data):
        # anchor1 = raw_format["anchors"][0]
        # anchor2 = raw_format["anchors"][1]
        # anchor3 = raw_format["anchors"][2]
        # relation1 = raw_format["relations"][0]
        # relation2 = raw_format["relations"][1]
        # relation3 = raw_format["relations"][2]
        # target = random.choice(processed_graph_data["edge_dict"][anchor1][relation1] & processed_graph_data["edge_dict"][anchor2][relation2] & processed_graph_data["edge_dict"][anchor3][relation3])
        #
        # return {0: target}
        return {0: random.sample(raw_format["targets"], 1)[0]}

    @staticmethod
    def sample_ip(raw_format, processed_graph_data):
        anchor1 = raw_format["anchors"][0]
        anchor2 = raw_format["anchors"][1]
        relation1 = raw_format["relations"][0]
        relation2 = raw_format["relations"][1]
        relation3 = raw_format["relations"][2]
        variable = random.sample(processed_graph_data["edge_dict"][anchor1][relation1] & processed_graph_data["edge_dict"][anchor2][relation2] & processed_graph_data["relation_dict"][relation3], 1)[0]
        target = random.sample(processed_graph_data["edge_dict"][variable][relation3], 1)[0]

        return {0: variable, 1: target}

    @staticmethod
    def sample_pi(raw_format, processed_graph_data):
        anchor1 = raw_format["anchors"][0]
        relation1 = raw_format["relations"][0]
        relation2 = raw_format["relations"][1]
        variable_candidates = set()
        for target in raw_format["targets"]:
            variable_candidates |= processed_graph_data["edge_dict"][target][return_opposite_relation(relation2)]

        variable = random.sample(processed_graph_data["edge_dict"][anchor1][relation1] & variable_candidates, 1)[0]
        target = random.sample(processed_graph_data["edge_dict"][variable][relation2], 1)[0]

        return {0: variable, 1: target}

sample_subgraph_path = getattr(Subgraph_Path_Sampler, "sample_path")

class Subgraph_Reformatter(object):

    @staticmethod
    def reformat_subgraph(raw_format, query_structure, loaded_graph_data, processed_graph_data):
        method_name = 'reformat_' + str(query_structure)
        method = getattr(Subgraph_Reformatter, method_name, Subgraph_Reformatter.handle_error)
        return method(raw_format, loaded_graph_data, processed_graph_data)

    @staticmethod
    def handle_error(raw_format):
        print("Invalid Query Structure")
        return None

    @staticmethod
    def reformat_1p(raw_format, loaded_graph_data, processed_graph_data):
        relation_index = return_opposite_relation(raw_format["relations"][0])
        relation = loaded_graph_data["relation_index"][relation_index]

        sampled_variables = sample_subgraph_path(raw_format, "1p", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["1p"]]
        query_graph.append((sampled_variables[0], relation, raw_format["anchors"][0]))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        for edge in query_graph[1:]:
            rels.append(edge[1])
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("1-chain", rels)


    @staticmethod
    def reformat_2p(raw_format, loaded_graph_data, processed_graph_data):
        relation_index1 = return_opposite_relation(raw_format["relations"][0])
        relation_index2 = return_opposite_relation(raw_format["relations"][1])
        relation1 = loaded_graph_data["relation_index"][relation_index1]
        relation2 = loaded_graph_data["relation_index"][relation_index2]

        sampled_variables = sample_subgraph_path(raw_format, "2p", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["2p"]]
        query_graph.append((sampled_variables[1], relation2, sampled_variables[0]))
        query_graph.append((sampled_variables[0], relation1, raw_format["anchors"][0]))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        for edge in query_graph[1:]:
            rels.append(edge[1])
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("2-chain", rels)


    @staticmethod
    def reformat_3p(raw_format, loaded_graph_data, processed_graph_data):
        relation_index1 = return_opposite_relation(raw_format["relations"][0])
        relation_index2 = return_opposite_relation(raw_format["relations"][1])
        relation_index3 = return_opposite_relation(raw_format["relations"][2])
        relation1 = loaded_graph_data["relation_index"][relation_index1]
        relation2 = loaded_graph_data["relation_index"][relation_index2]
        relation3 = loaded_graph_data["relation_index"][relation_index3]

        sampled_variables = sample_subgraph_path(raw_format, "3p", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["3p"]]
        query_graph.append((sampled_variables[2], relation3, sampled_variables[1]))
        query_graph.append((sampled_variables[1], relation2, sampled_variables[0]))
        query_graph.append((sampled_variables[0], relation1, raw_format["anchors"][0]))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        for edge in query_graph[1:]:
            rels.append(edge[1])
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("3-chain", rels)

    @staticmethod
    def reformat_2i(raw_format, loaded_graph_data, processed_graph_data):
        relation_index1 = return_opposite_relation(raw_format["relations"][0])
        relation_index2 = return_opposite_relation(raw_format["relations"][1])
        relation1 = loaded_graph_data["relation_index"][relation_index1]
        relation2 = loaded_graph_data["relation_index"][relation_index2]

        sampled_variables = sample_subgraph_path(raw_format, "2i", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["2i"]]
        query_graph.append((sampled_variables[0], relation2, raw_format["anchors"][1]))
        query_graph.append((sampled_variables[0], relation1, raw_format["anchors"][0]))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        for edge in query_graph[1:]:
            rels.append(edge[1])
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("2-inter", rels)

    @staticmethod
    def reformat_3i(raw_format, loaded_graph_data, processed_graph_data):
        relation_index1 = return_opposite_relation(raw_format["relations"][0])
        relation_index2 = return_opposite_relation(raw_format["relations"][1])
        relation_index3 = return_opposite_relation(raw_format["relations"][2])
        relation1 = loaded_graph_data["relation_index"][relation_index1]
        relation2 = loaded_graph_data["relation_index"][relation_index2]
        relation3 = loaded_graph_data["relation_index"][relation_index3]

        sampled_variables = sample_subgraph_path(raw_format, "3i", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["3i"]]
        query_graph.append((sampled_variables[0], relation3, raw_format["anchors"][2]))
        query_graph.append((sampled_variables[0], relation2, raw_format["anchors"][1]))
        query_graph.append((sampled_variables[0], relation1, raw_format["anchors"][0]))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        for edge in query_graph[1:]:
            rels.append(edge[1])
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("3-inter", rels)

    @staticmethod
    def reformat_ip(raw_format, loaded_graph_data, processed_graph_data):
        relation_index1 = return_opposite_relation(raw_format["relations"][0])
        relation_index2 = return_opposite_relation(raw_format["relations"][1])
        relation_index3 = return_opposite_relation(raw_format["relations"][2])
        relation1 = loaded_graph_data["relation_index"][relation_index1]
        relation2 = loaded_graph_data["relation_index"][relation_index2]
        relation3 = loaded_graph_data["relation_index"][relation_index3]

        sampled_variables = sample_subgraph_path(raw_format, "ip", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["ip"]]
        query_graph.append((sampled_variables[1], relation3, sampled_variables[0]))
        query_graph.append(((sampled_variables[0], relation2, raw_format["anchors"][1]), (sampled_variables[0], relation1, raw_format["anchors"][0])))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        rels.append(query_graph[1][1])
        rels.append((query_graph[2][0][1], (query_graph[2][1][1])))
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("3-chain_inter", rels)

    @staticmethod
    def reformat_pi(raw_format, loaded_graph_data, processed_graph_data):
        relation_index1 = return_opposite_relation(raw_format["relations"][0])
        relation_index2 = return_opposite_relation(raw_format["relations"][1])
        relation_index3 = return_opposite_relation(raw_format["relations"][2])
        relation1 = loaded_graph_data["relation_index"][relation_index1]
        relation2 = loaded_graph_data["relation_index"][relation_index2]
        relation3 = loaded_graph_data["relation_index"][relation_index3]

        sampled_variables = sample_subgraph_path(raw_format, "pi", processed_graph_data)

        query_graph = [loaded_graph_data["query_structure_index"]["pi"]]
        query_graph.append((sampled_variables[1], relation3, raw_format["anchors"][1]))
        query_graph.append(((sampled_variables[1], relation2, sampled_variables[0]), (sampled_variables[0], relation1, raw_format["anchors"][0])))
        query_graph = tuple(query_graph)

        query_data = (query_graph, raw_format["negs"], raw_format["hard_negs"])

        rels = []
        rels.append(query_graph[1][1])
        rels.append((query_graph[2][0][1], (query_graph[2][1][1])))
        rels = tuple(rels)

        return query_data, raw_format["anchors"], raw_format["targets"], Formula("3-inter_chain", rels)

class Validate_Test_Subgraph_Generator(object):

    @staticmethod
    def generate_subgraph_with_edge(query_structure, processed_graph_data, removed_indices):
        method_name = 'generate_' + str(query_structure)
        method = getattr(Validate_Test_Subgraph_Generator, method_name, Validate_Test_Subgraph_Generator.handle_error)
        return method(processed_graph_data, removed_indices)

    @staticmethod
    def handle_error(processed_graph_data):
        print("Invalid Query Structure")
        return None

    @staticmethod
    def generate_1p(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        return Subgraph_Generator.generate_subgraph("1p", processed_graph_data, first_int=first_int)

    @staticmethod
    def generate_2p(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        edge_index = random.randint(1,2)

        if edge_index == 1:
            return Subgraph_Generator.generate_subgraph("2p", processed_graph_data, first_int=first_int, removed_indices=removed_indices)
        else:
            trace = Back_Tracer.generate_trace("2p", processed_graph_data, first_int)
            if not trace:
                return Validate_Test_Subgraph_Generator.generate_2p(processed_graph_data, removed_indices)
            else:
                return Subgraph_Generator.generate_subgraph("2p", processed_graph_data, trace=trace, removed_indices=removed_indices)

    @staticmethod
    def generate_3p(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        edge_index = random.randint(1,3)

        if edge_index == 1:
            return Subgraph_Generator.generate_subgraph("3p", processed_graph_data, first_int=first_int, removed_indices=removed_indices)
        else:
            trace = Back_Tracer.generate_trace("3p", processed_graph_data, first_int, edge_index)
            if not trace:
                return Validate_Test_Subgraph_Generator.generate_3p(processed_graph_data, removed_indices)
            else:
                return Subgraph_Generator.generate_subgraph("3p", processed_graph_data, trace=trace, removed_indices=removed_indices)

    @staticmethod
    def generate_2i(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        return Subgraph_Generator.generate_subgraph("2i", processed_graph_data, first_int=first_int, removed_indices=removed_indices)

    @staticmethod
    def generate_3i(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        return Subgraph_Generator.generate_subgraph("3i", processed_graph_data, first_int=first_int, removed_indices=removed_indices)

    @staticmethod
    def generate_ip(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        edge_index = random.randint(1,3)

        if edge_index == 1 or edge_index == 2:
            return Subgraph_Generator.generate_subgraph("ip", processed_graph_data, first_int=first_int, removed_indices=removed_indices)
        else:
            trace = Back_Tracer.generate_trace("ip", processed_graph_data, first_int)
            
            if not trace:
                return Validate_Test_Subgraph_Generator.generate_ip(processed_graph_data, removed_indices)
            else:
                return Subgraph_Generator.generate_subgraph("ip", processed_graph_data, trace=trace, removed_indices=removed_indices)

    @staticmethod
    def generate_pi(processed_graph_data, removed_indices):
        first_int = random.choice(removed_indices)
        edge_index = random.randint(1,3)

        if edge_index == 1:
            return Subgraph_Generator.generate_subgraph("pi", processed_graph_data, first_int=first_int, removed_indices=removed_indices)
        else:
            trace = Back_Tracer.generate_trace("pi", processed_graph_data, first_int, edge_index)
            if not trace:
                return Validate_Test_Subgraph_Generator.generate_pi(processed_graph_data, removed_indices)
            else:
                return Subgraph_Generator.generate_subgraph("pi", processed_graph_data, trace=trace, removed_indices=removed_indices)

class Back_Tracer(object):

    @staticmethod
    def generate_trace(query_structure, processed_graph_data, first_int, edge_index=None):
        method_name = 'generate_' + str(query_structure)
        method = getattr(Back_Tracer, method_name, Back_Tracer.handle_error)
        if not edge_index:
            return method(processed_graph_data, first_int)
        return method(processed_graph_data, first_int, edge_index)

    @staticmethod
    def handle_error(processed_graph_data):
        print("Invalid Query Structure")
        return None

    @staticmethod
    def generate_2p(processed_graph_data, first_int):
        relation = processed_graph_data["edge_type_dict"][first_int]
        var_node = processed_graph_data["edge_index_dict"][0][first_int]
        tar_node = processed_graph_data["edge_index_dict"][1][first_int]
        
        candidates = processed_graph_data["edge_indices_dict"][var_node]
        first_edge = random.sample(candidates, 1)[0]
        anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
        first_relation = processed_graph_data["edge_type_dict"][first_edge]
        if first_relation == relation and anchor_node == tar_node:
            candidates = candidates - {first_edge}
            if not candidates:
                return None
            else:
                first_edge = random.sample(candidates, 1)[0]
                anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
                first_relation = processed_graph_data["edge_type_dict"][first_edge]
        
        return [[anchor_node], [return_opposite_relation(first_relation), relation]]

    @staticmethod
    def generate_3p(processed_graph_data, first_int, edge_index=None):
        if edge_index == 2:
            relation = processed_graph_data["edge_type_dict"][first_int]
            var_node = processed_graph_data["edge_index_dict"][0][first_int]
            tar_node = processed_graph_data["edge_index_dict"][1][first_int]

            candidates = processed_graph_data["edge_indices_dict"][var_node]
            first_edge = random.sample(candidates, 1)[0]
            anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
            first_relation = processed_graph_data["edge_type_dict"][first_edge]
            if first_relation == relation and anchor_node == tar_node:
                candidates = candidates - {first_edge}
                if not candidates:
                    return None
                else:
                    first_edge = random.sample(candidates, 1)[0]
                    anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
                    first_relation = processed_graph_data["edge_type_dict"][first_edge]

            return [[anchor_node], [return_opposite_relation(first_relation), relation]]
        elif edge_index == 3:
            relation = processed_graph_data["edge_type_dict"][first_int]
            var_node1 = processed_graph_data["edge_index_dict"][0][first_int]
            tar_node = processed_graph_data["edge_index_dict"][1][first_int]

            candidates1 = processed_graph_data["edge_indices_dict"][var_node1]
            second_edge = random.sample(candidates1, 1)[0]
            var_node2 = processed_graph_data["edge_index_dict"][1][second_edge]
            second_relation = processed_graph_data["edge_type_dict"][second_edge]
            if second_relation == relation and var_node2 == tar_node:
                candidates1 = candidates1 - {second_edge}
                if not candidates1:
                    return None
                else:
                    second_edge = random.sample(candidates1, 1)[0]
                    var_node2 = processed_graph_data["edge_index_dict"][1][second_edge]
                    second_relation = processed_graph_data["edge_type_dict"][second_edge]

            candidates2 = processed_graph_data["edge_indices_dict"][var_node2]
            first_edge = random.sample(candidates2, 1)[0]
            anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
            first_relation = processed_graph_data["edge_type_dict"][first_edge]
            if first_edge == return_opposite_relation(second_relation) and anchor_node == var_node2:
                candidates2 = candidates2 - {first_edge}
                if not candidates2:
                    return None
                else:
                    first_edge = random.sample(candidates2, 1)[0]
                    anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
                    first_relation = processed_graph_data["edge_type_dict"][first_edge]

            return [[anchor_node], [return_opposite_relation(first_relation), return_opposite_relation(second_relation), relation]]
        else:
            raise ValueError("Edge index should be 2 or 3 for 3p.")

    @staticmethod
    def generate_ip(processed_graph_data, first_int):
        relation = processed_graph_data["edge_type_dict"][first_int]
        var_node = processed_graph_data["edge_index_dict"][0][first_int]
        tar_node = processed_graph_data["edge_index_dict"][1][first_int]

        candidates = processed_graph_data["edge_indices_dict"][var_node]
        first_edge = random.sample(candidates, 1)[0]
        anchor_node1 = processed_graph_data["edge_index_dict"][1][first_edge]
        first_relation = processed_graph_data["edge_type_dict"][first_edge]
        if first_relation == relation and anchor_node1 == tar_node:
            candidates = candidates - {first_edge}
            if not candidates:
                return None
            else:
                first_edge = random.sample(candidates, 1)[0]
                anchor_node1 = processed_graph_data["edge_index_dict"][1][first_edge]
                first_relation = processed_graph_data["edge_type_dict"][first_edge]

        second_edge = random.sample(candidates, 1)[0]
        anchor_node2 = processed_graph_data["edge_index_dict"][1][second_edge]
        second_relation = processed_graph_data["edge_type_dict"][second_edge]
        if (second_relation == relation and anchor_node2 == tar_node) or (second_relation == first_relation and anchor_node2 == anchor_node1):
            candidates = candidates - {second_edge}
            if not candidates:
                return None
            else:
                second_edge = random.sample(candidates, 1)[0]
                anchor_node2 = processed_graph_data["edge_index_dict"][1][second_edge]
                second_relation = processed_graph_data["edge_type_dict"][second_edge]

        return [[anchor_node1, anchor_node2], [return_opposite_relation(first_relation), return_opposite_relation(second_relation), relation]]

    @staticmethod
    def generate_pi(processed_graph_data, first_int, edge_index=None):
        if edge_index == 2:
            relation = processed_graph_data["edge_type_dict"][first_int]
            var_node = processed_graph_data["edge_index_dict"][0][first_int]
            tar_node = processed_graph_data["edge_index_dict"][1][first_int]

            candidates = processed_graph_data["edge_indices_dict"][var_node]
            first_edge = random.sample(candidates, 1)[0]
            anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
            first_relation = processed_graph_data["edge_type_dict"][first_edge]
            if first_relation == relation and anchor_node == tar_node:
                candidates = candidates - {first_edge}
                if not candidates:
                    return None
                else:
                    first_edge = random.sample(candidates, 1)[0]
                    anchor_node = processed_graph_data["edge_index_dict"][1][first_edge]
                    first_relation = processed_graph_data["edge_type_dict"][first_edge]

            return [[anchor_node], [return_opposite_relation(first_relation), relation]]
        elif edge_index == 3:
            relation = processed_graph_data["edge_type_dict"][first_int]
            tar_node = processed_graph_data["edge_index_dict"][1][first_int]
            anchor_node2 = processed_graph_data["edge_index_dict"][0][first_int]

            candidates1 = processed_graph_data["edge_indices_dict"][tar_node]
            second_edge = random.sample(candidates1, 1)[0]
            var_node = processed_graph_data["edge_index_dict"][1][second_edge]
            second_relation = processed_graph_data["edge_type_dict"][second_edge]

            if second_relation == return_opposite_relation(relation) and tar_node == anchor_node2:
                candidates1 = candidates1 - {second_edge}
                if not candidates1:
                    return None
                else:
                    second_edge = random.sample(candidates1, 1)[0]
                    var_node = processed_graph_data["edge_index_dict"][1][second_edge]
                    second_relation = processed_graph_data["edge_type_dict"][second_edge]

            candidates2 = processed_graph_data["edge_indices_dict"][var_node]
            first_edge = random.sample(candidates2, 1)[0]
            anchor_node1 = processed_graph_data["edge_index_dict"][1][first_edge]
            first_relation = processed_graph_data["edge_type_dict"][first_edge]
            if first_edge == return_opposite_relation(second_relation) and anchor_node1 == var_node:
                candidates2 = candidates2 - {first_edge}
                if not candidates2:
                    return None
                else:
                    first_edge = random.sample(candidates2, 1)[0]
                    anchor_node1 = processed_graph_data["edge_index_dict"][1][first_edge]
                    first_relation = processed_graph_data["edge_type_dict"][first_edge]

            return [[anchor_node1, anchor_node2], [return_opposite_relation(first_relation), return_opposite_relation(second_relation), relation]]
        else:
            raise ValueError("Edge index should be 2 or 3 for pi.")

class ruud_RGCNQueryDataset(QueryDataset):
    """A dataset for queries of a specific type, e.g. 1-chain.
    The dataset contains queries for formulas of different types, e.g.
    200 queries of type (('protein', '0', 'protein')),
    500 queries of type (('protein', '0', 'function')).
    (note that these queries are of type 1-chain).

    Args:
        queries (dict): maps formulas (graph.Formula) to query instances
            (list of graph.Query?)
    """
    query_edge_indices = {'1-chain': [[0],
                                      [1]],
                          '2-chain': [[0, 2],
                                      [2, 1]],
                          '3-chain': [[0, 3, 2],
                                      [3, 2, 1]],
                          '2-inter': [[0, 1],
                                      [2, 2]],
                          '3-inter': [[0, 1, 2],
                                      [3, 3, 3]],
                          '3-inter_chain': [[0, 1, 3],
                                            [2, 3, 2]],
                          '3-chain_inter': [[0, 1, 3],
                                            [3, 3, 2]]}

    query_diameters = {'1-chain': 1,
                       '2-chain': 2,
                       '3-chain': 3,
                       '2-inter': 1,
                       '3-inter': 1,
                       '3-inter_chain': 2,
                       '3-chain_inter': 2}

    query_edge_label_idx = {'1-chain': [0],
                            '2-chain': [1, 0],
                            '3-chain': [2, 1, 0],
                            '2-inter': [0, 1],
                            '3-inter': [0, 1, 2],
                            '3-inter_chain': [0, 2, 1],
                            '3-chain_inter': [1, 2, 0]}

    variable_node_idx = {'1-chain': [0],
                         '2-chain': [0, 2],
                         '3-chain': [0, 2, 4],
                         '2-inter': [0],
                         '3-inter': [0],
                         '3-chain_inter': [0, 2],
                         '3-inter_chain': [0, 3]}

    def __init__(self, queries, enc_dec):
        super(ruud_RGCNQueryDataset, self).__init__(queries)
        self.mode_ids = enc_dec.mode_ids
        self.rel_ids = enc_dec.rel_ids

    def collate_fn(self, idx_list):
        formula, queries = super(ruud_RGCNQueryDataset, self).collate_fn(idx_list)
        graph_data = ruud_RGCNQueryDataset.get_query_graph(formula, queries,
                                                      self.rel_ids,
                                                      self.mode_ids)
        anchor_ids, var_ids, graph = graph_data
        return formula, queries, anchor_ids, var_ids, graph

    @staticmethod
    def get_query_graph(formula, queries, rel_ids, mode_ids):
        batch_size = len(queries)
        n_anchors = len(formula.anchor_modes)

        # The rest of the rows contain generic mode embeddings for variables
        all_nodes = formula.get_nodes()
        var_idx = ruud_RGCNQueryDataset.variable_node_idx[formula.query_type]
        var_ids = np.array([mode_ids[all_nodes[i]] for i in var_idx],
                            dtype=np.int)

        edge_index = ruud_RGCNQueryDataset.query_edge_indices[formula.query_type]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        rels = formula.get_rels()
        rel_idx = ruud_RGCNQueryDataset.query_edge_label_idx[formula.query_type]
        edge_type = [rel_ids[_reverse_relation(rels[i])] for i in rel_idx]
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        edge_data = Data(edge_index=edge_index)
        edge_data.edge_type = edge_type
        edge_data.num_nodes = n_anchors + len(var_idx)

        graph = Batch.from_data_list([edge_data for i in range(batch_size)])

        return (torch.tensor(var_ids, dtype=torch.long),
                graph)

def create_initial_embeddings(num_nodes, emb_dim, device=None):
    uniform_distribution = Uniform(torch.tensor([-100.0]), torch.tensor([100.0]))
    normal_distribution = Normal(torch.tensor([5.0]), torch.tensor([2.0]))

    centers = uniform_distribution.sample((num_nodes, int(emb_dim / 2))).squeeze(-1)
    offsets = normal_distribution.sample((num_nodes, int(emb_dim / 2))).squeeze(-1)

    node_embeddings = torch.cat((centers, offsets), 1).detach()
    if device:
        node_embeddings = node_embeddings.to(device)
    node_embeddings.requires_grad = True

    return node_embeddings

def return_downsampled_graph(edge_index, edge_type):
    downsampled_index = edge_index.tolist()
    downsampled_type = edge_type.tolist()
    remove_amount = round(len(edge_index[0]) / 10)
    to_remove = sorted(random.sample(range(edge_index.shape[1]), remove_amount), reverse=True)

    removed_index = [[], []]
    removed_type = []

    for index in to_remove:
        removed_index[0].insert(0, downsampled_index[0][index])
        removed_index[1].insert(0, downsampled_index[1][index])
        removed_type.insert(0, downsampled_type[index])

        del downsampled_index[0][index]
        del downsampled_index[1][index]
        del downsampled_type[index]

    return torch.tensor(downsampled_index, dtype=torch.long, requires_grad=False), torch.tensor(downsampled_type, dtype=torch.long, requires_grad=False), torch.tensor(removed_index, dtype=torch.long, requires_grad=False), torch.tensor(removed_type, dtype=torch.long, requires_grad=False)

def remove_query_graph_edge(q_graphs):
    rand_index = random.randint(0, q_graphs.edge_index.shape[1] - 1)
    q_graphs.removed_edge_index = torch.stack(
        (q_graphs.edge_index[0][rand_index].reshape((1)), q_graphs.edge_index[1][rand_index].reshape((1))))
    q_graphs.removed_edge_type = q_graphs.edge_type[rand_index].reshape((1))

    if q_graphs.edge_index.shape[1] == 1:
        q_graphs.edge_index = torch.empty((2, 0), dtype=torch.long, requires_grad=False)
        q_graphs.edge_type = torch.empty((0), dtype=torch.long, requires_grad=False)
    else:
        if rand_index == 0:
            q_graphs.edge_index = torch.stack((q_graphs.edge_index[0][1:], q_graphs.edge_index[1][1:]))
            q_graphs.edge_type = q_graphs.edge_type[1:]
        elif rand_index == q_graphs.edge_index.shape[1] - 1:
            q_graphs.edge_index = torch.stack(
                (q_graphs.edge_index[0][:rand_index], q_graphs.edge_index[1][:rand_index]))
            q_graphs.edge_type = q_graphs.edge_type[:rand_index]
        else:
            q_graphs.edge_index = torch.stack((torch.cat(
                (q_graphs.edge_index[0][:rand_index], q_graphs.edge_index[0][rand_index + 1:])), torch.cat(
                (q_graphs.edge_index[1][:rand_index], q_graphs.edge_index[1][rand_index + 1:]))))
            q_graphs.edge_type = torch.cat((q_graphs.edge_type[:rand_index], q_graphs.edge_type[rand_index + 1:]))

def get_query_structures(query):
    raw_query = query["raw_query"]
    query_type = query["query"].formula.query_type

    anchor_str = "a:_"
    relation_str = "r:_"

    if query_type == "2-inter":  # 2i
        relation_str1 = relation_str + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][1])
        relation_str2 = relation_str + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][0])

        structures = {anchor_str + relation_str1, anchor_str + relation_str2}
    elif query_type == "3-inter":  # 3i
        relation_str1 = relation_str + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][2])
        relation_str2 = relation_str + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][2]) + "_" + str(raw_query["relations"][1])
        relation_str3 = relation_str + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][2])
        relation_str4 = relation_str + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][2]) + "_" + str(raw_query["relations"][0])
        relation_str5 = relation_str + str(raw_query["relations"][2]) + "_" + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][1])
        relation_str6 = relation_str + str(raw_query["relations"][2]) + "_" + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][0])

        structures = {anchor_str + relation_str1, anchor_str + relation_str2, anchor_str + relation_str3,
                      anchor_str + relation_str4, anchor_str + relation_str5, anchor_str + relation_str6}
    elif query_type == "3-chain_inter":  # ip
        relation_str1 = relation_str + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][2])
        relation_str2 = relation_str + str(raw_query["relations"][1]) + "_" + str(raw_query["relations"][0]) + "_" + str(raw_query["relations"][2])

        structures = {anchor_str + relation_str1, anchor_str + relation_str2}
    else:
        for i in range(len(raw_query["anchors"])):
            anchor_str += str(raw_query["anchors"][i]) + "_"

        for i in range(len(raw_query["relations"])):
            relation_str += str(raw_query["relations"][i]) + "_"

        relation_str = relation_str[:-1]
        structures = {anchor_str + relation_str}
    return structures
