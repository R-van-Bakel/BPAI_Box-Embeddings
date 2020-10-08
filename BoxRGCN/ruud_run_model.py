import Github_downloads.mpqe_master.mpqe.data_utils as data_utils
import Github_downloads.mpqe_master.mpqe.encoders as encoders
from ruud_MPQE import ruud_MPQE
import pickle

graph, feature_modules, node_maps = data_utils.load_graph("BoxRGCN_Data", 128)
features = graph.features

with open("BoxRGCN_Data/relations.pkl", "rb") as f:
    relations = pickle.load(f)

# encoder = encoders.DirectEncoder(features, feature_modules)
# my_model = model.RGCNEncoderDecoder(graph=graph, enc=encoder, readout="sum", scatter_op="add", dropout=0, weight_decay=0, num_layers=2, shared_layers=False, adaptive=True)

my_model = ruud_MPQE(128, len(relations))

my_model()


# my_model.forward()
# print(my_model.rel_ids)
# print(my_model.mode_ids)
