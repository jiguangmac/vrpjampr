from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS, RPEnv
import torch
from torch.utils.data import DataLoader
from lib.model.policy import Policy
import logging
from lib.utils.challenge_utils import dimacs_challenge_dist_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
LPATH = "./data/mydata_prep.pkl"
SMP = 1
N = 13
BS = 1
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

env = RPEnv(debug=True, device=device, k_nbh_frac=0.25)
env.seed(SEED+1)

model = Policy(
    observation_space=env.OBSERVATION_SPACE,
    embedding_dim=128,
)

for batch in dl:
    model.reset_static()
    env.load_data(batch)
    obs = env.reset()
    #start
    """
    nbh = env.get_node_nbh(env.cur_node)
    print("nbh",nbh)
    mask = (env._visited[:,None,:].expand(env.bs,env.max_concurrent_vehicles,-1).gather(dim=-1,index=nbh))
    print("mask",mask)
    exceeds_cap = env.cur_cap[:, :, None] < env.demands[:, None, :].expand(env.bs,env.max_concurrent_vehicles,-1).gather(dim=-1,index=nbh)
    print("exceeds_cap",exceeds_cap)
    idx_pair = torch.stack((env.cur_node[:, :, None].expand(env.bs, env.max_concurrent_vehicles,env.k_nbh_size).reshape(env.bs,-1),nbh.view(env.bs,-1)),dim=-1).view(env.bs,-1)
    print("idx_pair",idx_pair)
    idx_coords = env.coords.gather(dim=1,index=idx_pair[:,:,None].expand(env.bs,-1,2)).view(env.bs,-1,2,2)
    print("idx_coords",idx_coords)
    print("org",env.org_service_horizon)
    t_delta = (dimacs_challenge_dist_fn(idx_coords[:, :, 0, :], idx_coords[:, :, 1, :]) / env.org_service_horizon[env._bidx][:,None]).view(env.bs,env.max_concurrent_vehicles,-1)
    print(t_delta)
    arrival_time = env.cur_time[:,:,None] + t_delta
    exceeds_tw = arrival_time >(env.tw[:,:,-1][:,None,:].expand(env.bs,env.max_concurrent_vehicles,env.graph_size).gather(dim=-1,index=nbh))
    print("exceeds_tw",exceeds_tw)
    mask_depot = ~env.visited.all(-1) & ((env._finished.sum(-1) < env.max_concurrent_vehicles) | (env._get_next_active_vehicle() != 0))
    at_depot = env.cur_node == 0
    mask_depot = at_depot & mask_depot[:,None].expand(-1,env.max_concurrent_vehicles)
    print(mask_depot)
    if mask_depot.any():
        mask[mask_depot, torch.zeros(mask_depot.sum(), dtype=torch.long)] = 1
        print(mask)
    #end
    """
    i = 0
    done = False
    while not done:
        action, log_likelihood, entropy = model(obs, recompute_static=(i==0))
        print("action",action)
        obs, rew, done, info = env.step(action)
        print("obs",obs)
        i += 1

    print(info)
