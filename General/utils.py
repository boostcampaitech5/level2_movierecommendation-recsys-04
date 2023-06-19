import os
import sys

import importlib
import argparse
import yaml

import random

import numpy as np
import pandas as pd

import torch

sys.path.append("./model")


def load_config(args):
    print("##### Load config ...")
    with open(f"./config/{args.model}.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = config[f"Ver{args.config_ver}"]
        print(config)

    return config


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def get_model(config):
    try:
        model = importlib.import_module(config["model"])
        model = getattr(model, config["model"])
        return model

    except ImportError:
        print(f"Module '{config['model']}' not found")
    except AttributeError:
        print(
            f"Class '{config['model']}' not found in module '{config['model']}'"
        )


def save_result(config, preds):
    os.makedirs(f"./output", exist_ok=True)
    os.makedirs(f"./output/{config['model']}", exist_ok=True)
    preds.to_csv(
        f"./output/{config['model']}/{config['model']}_ver_{config['config_ver']}_submission_with_item_scores.csv",
        index=False,
    )
    preds = preds.drop(columns="score")
    preds.to_csv(
        f"./output/{config['model']}/{config['model']}_ver_{config['config_ver']}_submission.csv",
        index=False,
    )
