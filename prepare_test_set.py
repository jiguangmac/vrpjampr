import os
import pickle

from lib.utils.io_utils import (
    read_tsplib_cvrptw,
    normalize_instance,
    to_rp_instance,
)

LPATH = "./data/solomon_txt/"
DATA_SPATH = "./data/solomon_prep.pkl"
GROUPS = ["r", "c", "rc"]
TYPES = ["1", "2"]

instances = {}
for g in GROUPS:
    types = [f"{g}{t}" for t in TYPES]
    for type in types:
        print(f"processing type: {type}")
        load_pth = os.path.join(LPATH, type)
        file_names = os.listdir(load_pth)
        file_names.sort()

        data = []
        for fname in file_names:
            pth = os.path.join(load_pth, fname)
            print(f"preparing file '{fname}' from {pth}")
            instance = read_tsplib_cvrptw(pth)
            instance = normalize_instance(instance)
            data.append(instance)
            print(instance)
        buffer = {'tw_frac=1.0': []}
        # infer tw frac of instance
        for instance in data:
            org_df = instance['features'].loc[1:, :]    # without depot!
            has_tw = (org_df.tw_start != 0)
            tw_frac = has_tw.sum() / org_df.shape[0]
            buffer[f"tw_frac={tw_frac}"].append(to_rp_instance(instance))

        instances[type] = buffer


with open(DATA_SPATH, 'wb') as f:
    pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)
