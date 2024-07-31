from lib.routing import RPDataset, GROUPS, TYPES, TW_FRACS, RPEnv

from lib.model.policy import Policy

SAMPLE_CFG = {"groups": GROUPS, "types": TYPES, "tw_fracs": TW_FRACS}
LPATH = "./solomon_stats.pkl"
SMP = 9
N = 20
BS = 3
MAX_CON = 3
CUDA = False
SEED = 123

device = torch.device("cuda" if CUDA else "cpu")
torch.manual_seed(SEED-1)

ds = RPDataset(cfg=SAMPLE_CFG, stats_pth=LPATH)
ds.seed(SEED)
data = ds.sample(sample_size=SMP, graph_size=N)

dl = DataLoader(
    data,
    batch_size=BS,
    collate_fn=lambda x: x,  # identity -> returning simple list of instances
    shuffle=False
)

env = RPEnv(debug=True, device=device, max_concurrent_vehicles=MAX_CON, k_nbh_frac=0.4)
env.seed(SEED+1)

model = Policy(
    observation_space=env.OBSERVATION_SPACE,
    embedding_dim=128,
)

for batch in dl:
    model.reset_static()
    env.load_data(batch)
    obs = env.reset()
    done = False
    i = 0

    while not done:
        action, log_likelihood, entropy = model(obs, recompute_static=(i==0))
        print("action",action)
        obs, rew, done, info = env.step(action)
        print("obs",obs)
        i += 1
        break

    print(info)
