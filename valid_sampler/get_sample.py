import os
import numpy as np
import pandas as pd

import argparse


def main(args):
    print("##### Load data ...")
    sota_top100 = pd.read_csv("./submissions/HighOrderEASE_top100.csv")
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

    valid_set = sampling(seed, sota_top100)

    os.makedirs("./output", exist_ok=True)
    valid_set.to_csv(f"./output/valid_set.csv", index=False)


def sampling(seed, sota, k=10):
    np.random.seed(seed)

    # 그룹화된 데이터프레임 생성
    grouped = sota.groupby("user")

    # 각 그룹에서 임의의 인덱스 선택
    shuffled_indices = grouped.apply(
        lambda x: np.random.choice(x.index, k, replace=False)
    )

    # 선택된 인덱스를 사용하여 해당 행을 가져옴
    result = sota.loc[shuffled_indices.explode()]

    # 결과 데이터프레임 생성
    result = result.reset_index(drop=True)

    return result


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
