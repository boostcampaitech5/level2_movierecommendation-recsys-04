import os
import natsort
import numpy as np
import pandas as pd

from tqdm import tqdm

import argparse


def main(args):
    global all_files_columns
    print("##### Load data ...")
    inters, submissions, all_files_path = load_data()
    all_files_columns = [file[:4] for file in all_files_path]
    print("##### Load data done!")

    print("##### Start searching seed ...")
    np.random.seed(args.start_seed)
    seeds = np.random.randint(0, 10000, args.n_iters)

    valid_result = pd.DataFrame(
        np.zeros((args.n_iters, len(all_files_columns) + 1)),
        columns=["seed"] + all_files_columns,
    )
    for i in tqdm(
        range(args.n_iters),
        desc="running ...",
    ):
        seed = seeds[i]
        recall_list = []
        recall_list.append(run(seed, inters, submissions))
        valid_result.loc[i, "seed"] = seed

        for j in range(len(all_files_columns)):
            valid_result.iloc[i, j + 1] = recall_list[0][j]

    print("##### Searching done!")
    valid_result["use_valid"] = False

    valid_result = valid_result.apply(lambda x: check_use_valid(x), axis=1)
    os.makedirs("./output", exist_ok=True)
    valid_result.to_csv(
        f"./output/valid_result_iter_{args.n_iters}.csv", index=False
    )
    seed = valid_result.loc[valid_result["use_valid"] == True, "seed"].values
    if len(seed) != 0:
        print("##### You must use these seed:", seed)
    else:
        print("##### Failed experiment. Retry increasing number of iterations")


def check_use_valid(x):
    global all_files_columns
    is_sorted = all(
        x.iloc[i + 1] <= x.iloc[i + 2]
        for i in range(len(all_files_columns) - 1)
    )
    x["use_valid"] = is_sorted

    return x


def sampling(seed, inters, k=10):
    np.random.seed(seed)

    # 그룹화된 데이터프레임 생성
    grouped = inters.groupby("user")

    # 각 그룹에서 임의의 인덱스 선택
    shuffled_indices = grouped.apply(
        lambda x: np.random.choice(x.index, k, replace=False)
    )

    # 선택된 인덱스를 사용하여 해당 행을 가져옴
    result = inters.loc[shuffled_indices.explode()]

    # 결과 데이터프레임 생성
    result = result.reset_index(drop=True)

    return result


def compute_recall(actual, predicted):
    actual = (
        actual.groupby("user")["item"].apply(list).reset_index()["item"].array
    )
    predicted = (
        predicted.groupby("user")["item"]
        .apply(list)
        .reset_index()["item"]
        .array
    )

    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0

    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1

    return round((sum_recall / true_users) * 100, 5)  # 비교를 편리하게 하기 위해 100 을 곱해줌


def run(seed, inters: pd.DataFrame, submissions: list):
    result = sampling(seed, inters)

    recalls = []
    for submission in submissions:
        recalls.append(compute_recall(result, submission))

    return recalls


def load_data():
    all_files_path = os.listdir("./submissions/")
    all_files_path = natsort.natsorted(all_files_path)

    submissions = []
    for file in tqdm(all_files_path):
        submissions.append(pd.read_csv("./submissions/" + file))

    # inters = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
    inters = pd.read_csv(
        "/opt/ml/level2_movierecommendation-recsys-04/ensemble/submissions/ensembles/soft-0.40-0.60(SASRec_sota_with_item_scores+HighOrderEASE_sota_with_item_scores).csv"
    )
    # inters = inters.drop(columns="time")
    return inters, submissions, all_files_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_iters", "-n", type=int, help="number of iterations"
    )
    parser.add_argument(
        "--start_seed", "-s", type=int, default=0, help="seed at start"
    )
    args = parser.parse_args()
    main(args)
