import os
import argparse
import numpy as np
import pandas as pd
import torch
from models import BERT4Rec
from utils import fix_random_seed
from preprocessing import preprocessing

import yaml
from tqdm import tqdm

# set config arument
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="0_1_0",
    help="input config version like '0_2_1'",
)
parser.add_argument(
    "--model_path",
    "-mp",
    type=str,
    default="",
    help="input model pth file path",
)
parser.add_argument("--topk", "-k", type=int, default=10)

args = parser.parse_args()

config_path = os.path.join("./configs/", f"Ver_{args.config}.yaml")

with open(config_path, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

# set random seed
fix_random_seed(config["seed"])

(
    user_ids,
    item_ids,
    num_user,
    num_item,
    users,
    user_train,
    user_valid,
) = preprocessing(config["data_path"])

model = torch.load(args.model_path).to(config["device"])

final = list()
cnt = 0

item_id = np.insert(item_ids, 0, 0)
tbar = tqdm(users.keys())

for step, key in enumerate(tbar):
    length = len(users[key])

    if length < config["max_len"]:
        dif = config["max_len"] - length
        data = [0] * dif + users[key][-length:]

    else:
        data = users[key][-config["max_len"] :]

    data = torch.LongTensor(data).unsqueeze(dim=0)
    result = model(data)[0]

    t = result[0].detach().cpu()
    t[data] = -np.inf
    top_k_idx = np.argpartition(t, -args.topk)[-args.topk :]
    rec_item_id = item_id[top_k_idx]
    rec_item_score = t[top_k_idx]
    user = user_ids[key]

    for item, score in zip(rec_item_id, rec_item_score):
        final.append((user, item, score.item()))

    tbar.set_description(f"step: {step:3d}")


info = pd.DataFrame(final, columns=["user", "item", "score"])
info.to_csv(os.path.join("./saved/", f'BERT4Rec_{config["version"]}.csv'))
print("Inferencd Done")
