import os
import importlib

import pandas as pd

from utils import *


def main(args):
    config = load_config(args)

    seed_everything(config["seed"])

    print("##### Load data ...")
    inters = pd.read_csv(config["data_path"])
    inters = inters.drop(columns="time")

    print("##### Initialize model ...")
    model = get_model(config)
    model = model(config["parameters"])

    print("##### Fit model ...")
    model.fit(inters)
    print("##### Fit Done!")

    users = inters["user"].unique()
    items = inters["item"].unique()

    print("##### Predict ...")
    preds = model.predict(inters, users, items, config["topk"])
    print("##### Predict done!")

    preds = preds.reset_index(drop=True)

    save_result(config, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, help="name of model to use (yaml)"
    )
    parser.add_argument(
        "--config_ver",
        "-c",
        type=str,
        default="0",
        help="veresion of experiments",
    )

    args = parser.parse_args()
    main(args)
