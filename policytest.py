from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS, RPEnv
import torch
from torch.utils.data import DataLoader
from lib.model.policy import Policy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
LPATH = "./data/mydata_prep.pkl"
SMP = 1
N = 13
BS = 1
MAX_CON = 3
CUDA = False
SEED = 123

device = torch.device("cuda" if CUDA else "cpu")
torch.manual_seed(SEED-1)

ds = RPDataset(cfg=SAMPLE_CFG, data_pth=LPATH)
ds.seed(SEED)
data = ds.sample(sample_size=SMP, graph_size=N)

dl = DataLoader(
    data,
    batch_size=BS,
    collate_fn=lambda x: x,  # identity -> returning simple list of instances
    shuffle=False
)

env = RPEnv(debug=True, device=device, max_concurrent_vehicles=MAX_CON, k_nbh_frac=0.25)
env.seed(SEED+1)

model = Policy(
    observation_space=env.OBSERVATION_SPACE,
    embedding_dim=128,
)

for batch in dl:
    model.reset_static()
    env.load_data(batch)
    obs = env.reset()
    i = 0
    done = False
    while not done:
        action, log_likelihood, entropy = model(obs, recompute_static=(i==0))
        print("action",action)
        obs, rew, done, info = env.step(action)
        print("obs",obs)
        i += 1

    print(info)
