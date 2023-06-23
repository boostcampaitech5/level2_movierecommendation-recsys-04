import os
import numpy as np
import pandas as pd

import argparse
from sampler import load_data, sampling


def main(args):
    print("##### Load data ...")
    inters, _, _ = load_data()
    print("##### Load data done!")

    np.random.seed(args.start_seed)
    seeds = np.random.randint(
        0, 10000, 200
    )  # 200 means "args.n_iter" in sampler.py
    seed = args.end_seed

    if seed in seeds:
        print("##### Seed success!")
    else:
        print("##### Something happens wrong...")

    valid_set = sampling(seed, inters)

    os.makedirs("./output", exist_ok=True)
    valid_set.to_csv(f"./output/valid_set.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--end_seed", "-e", type=int, default=2282, help="number of iterations"
    )
    parser.add_argument(
        "--start_seed", "-s", type=int, default=0, help="seed at start"
    )
    args = parser.parse_args()
    main(args)
