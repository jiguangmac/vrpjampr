import os
import pickle

from lib.utils.io_utils import (
    read_tsplib_cvrptw,
    my_normalize_instance,
    to_rp_instance,
)

LPATH = "./data/data_txt/"
DATA_SPATH = "./data/mydata_prep.pkl"

instances = {}
load_pth = os.path.join(LPATH)
file_name = os.listdir(load_pth)
data = []
pth = os.path.join(load_pth, file_name[0])
print(f"preparing file '{file_name}' from {pth}")
instance = read_tsplib_cvrptw(pth)
instance = my_normalize_instance(instance)
data.append(instance)
print(instance)
buffer = {'tw_frac=1.0': []}
# infer tw frac of instance
for instance in data:
    org_df = instance['features'].loc[1:, :]    # without depot!
    has_tw = (org_df.tw_start != 0)
    tw_frac = has_tw.sum() / org_df.shape[0]
    buffer[f"tw_frac={tw_frac}"].append(to_rp_instance(instance))

instances['r1'] = buffer

with open(DATA_SPATH, 'wb') as f:
    pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)
