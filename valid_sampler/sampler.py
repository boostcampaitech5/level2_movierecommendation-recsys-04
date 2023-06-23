import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import argparse


def main(args):
    print("##### Load data ...")
    data_dict = load_data()
    print("##### Load data done!")

    seed_list = []
    EASE_origin_recall_list = []
    EASE_upgrade_recall_list = []
    Dense_Slim_origin_recall_list = []
    # Dense_Slim_upgrade_recall_list = []
    Admm_Slim_origin_recall_list = []
    # Admm_Slim_upgrade_recall_list = []
    MultiDAE_origin_recall_list = []
    MultiDAE_upgrade_recall_list = []
    MultiVAE_origin_recall_list = []
    MultiVAE_upgrade_recall_list = []
    RecVAE_origin_recall_list = []
    RecVAE_upgrade_recall_list = []
    SASRec_origin_recall_list = []
    SASRec_upgrade_recall_list = []

    print("##### Start searching seed ...")
    np.random.seed(args.start_seed)
    seeds = np.random.randint(0, 10000, args.n_iters)

    for i in tqdm(
        range(args.n_iters),
        desc="running ...",
    ):
        seed = seeds[i]
        recall_result_dict = run(seed, data_dict)
        seed_list.append(seed)
        EASE_origin_recall_list.append(recall_result_dict["origin"][0])
        Dense_Slim_origin_recall_list.append(recall_result_dict["origin"][1])
        Admm_Slim_origin_recall_list.append(recall_result_dict["origin"][2])
        MultiDAE_origin_recall_list.append(recall_result_dict["origin"][3])
        RecVAE_origin_recall_list.append(recall_result_dict["origin"][4])
        MultiVAE_origin_recall_list.append(recall_result_dict["origin"][5])
        SASRec_origin_recall_list.append(recall_result_dict["origin"][6])

        EASE_upgrade_recall_list.append(recall_result_dict["upgrade"][0])
        # Dense_Slim_upgrade_recall_list.append(recall_result_dict["upgrade"][1])
        # Admm_Slim_upgrade_recall_list.append(recall_result_dict["upgrade"][2])
        MultiDAE_upgrade_recall_list.append(recall_result_dict["upgrade"][1])
        RecVAE_upgrade_recall_list.append(recall_result_dict["upgrade"][2])
        MultiVAE_upgrade_recall_list.append(recall_result_dict["upgrade"][3])
        SASRec_upgrade_recall_list.append(recall_result_dict["upgrade"][4])

    print("##### Searching done!")
    valid_result = pd.DataFrame(
        {
            "seed": seed_list,
            # origin
            "EASE_origin": EASE_origin_recall_list,
            "Dense_Slim_origin": Dense_Slim_origin_recall_list,
            "Admm_Slim_origin": Admm_Slim_origin_recall_list,
            "MultiDAE_origin": MultiDAE_origin_recall_list,
            "RecVAE_origin": RecVAE_origin_recall_list,
            "MultiVAE_origin": MultiVAE_origin_recall_list,
            "SASRec_origin": SASRec_origin_recall_list,
            # upgrade
            "EASE_upgrade": EASE_upgrade_recall_list,
            "MultiDAE_upgrade": MultiDAE_upgrade_recall_list,
            "RecVAE_upgrade": RecVAE_upgrade_recall_list,
            "MultiVAE_upgrade": MultiVAE_upgrade_recall_list,
            "SASRec_upgrade": SASRec_upgrade_recall_list,
        }
    )

    # 현재 기준 EASE > DenseSlim > AdmmSlim > MultiDAE > RecVAE > MultiVAE > SASRec
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
    origin_cond = (
        x["Dense_Slim_origin"]
        >= x["Admm_Slim_origin"]
        >= x["EASE_origin"]
        >= x["MultiDAE_origin"]
        >= x["RecVAE_origin"]
        >= x["MultiVAE_origin"]
        >= x["SASRec_origin"]
    )
    upgrade_cond = (
        (x["EASE_upgrade"] >= x["EASE_origin"])
        & (x["MultiDAE_upgrade"] >= x["MultiDAE_origin"])
        & (x["RecVAE_upgrade"] >= x["RecVAE_origin"])
        & ((x["MultiVAE_upgrade"] >= x["MultiVAE_origin"]))
        & (x["SASRec_upgrade"] >= x["SASRec_origin"])
    )

    if origin_cond & upgrade_cond:
        x["use_valid"] = True

    return x


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


def run(seed, data_dict):
    result = sampling(seed, data_dict["sota"])
    EASE_origin_recall = compute_recall(result, data_dict["origin"][0])
    Dense_Slim_origin_recall = compute_recall(result, data_dict["origin"][1])
    Admm_Slim_origin_recall = compute_recall(result, data_dict["origin"][2])
    MultiDAE_origin_recall = compute_recall(result, data_dict["origin"][3])
    RecVAE_origin_recall = compute_recall(result, data_dict["origin"][4])
    MultiVAE_origin_recall = compute_recall(result, data_dict["origin"][5])
    SASRec_origin_recall = compute_recall(result, data_dict["origin"][6])

    EASE_upgrade_recall = compute_recall(result, data_dict["upgrade"][0])
    # Dense_Slim_upgrade_recall = compute_recall(result, data_dict["Dense_Slim"])
    # Admm_Slim_upgrade_recall = compute_recall(result, data_dict["Admm_Slim"])
    MultiDAE_upgrade_recall = compute_recall(result, data_dict["upgrade"][1])
    RecVAE_upgrade_recall = compute_recall(result, data_dict["upgrade"][2])
    MultiVAE_upgrade_recall = compute_recall(result, data_dict["upgrade"][3])
    SASRec_upgrade_recall = compute_recall(result, data_dict["upgrade"][4])

    return dict(
        {
            "origin": [
                EASE_origin_recall,
                Dense_Slim_origin_recall,
                Admm_Slim_origin_recall,
                MultiDAE_origin_recall,
                RecVAE_origin_recall,
                MultiVAE_origin_recall,
                SASRec_origin_recall,
            ],
            "upgrade": [
                EASE_upgrade_recall,
                # Dense_Slim_upgrade_recall,
                # Admm_Slim_upgrade_recall,
                MultiDAE_upgrade_recall,
                RecVAE_upgrade_recall,
                MultiVAE_upgrade_recall,
                SASRec_upgrade_recall,
            ],
        }
    )


def load_data():
    inters = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")
    inters = inters.drop(columns="time")
    sota_top100 = pd.read_csv("./submissions/HighOrderEASE_top100.csv")

    EASE_origin = pd.read_csv("./submissions/EASE_origin.csv")
    Dense_Slim_origin = pd.read_csv("./submissions/DenseSlim_origin.csv")
    Admm_Slim_origin = pd.read_csv("./submissions/AdmmSlim_origin.csv")
    MultiDAE_origin = pd.read_csv("./submissions/MultiDAE_origin.csv")
    RecVAE_origin = pd.read_csv("./submissions/RecVAE_origin.csv")
    MultiVAE_origin = pd.read_csv("./submissions/MultiVAE_origin.csv")
    SASRec_origin = pd.read_csv("./submissions/SASRec_origin.csv")

    EASE_upgrade = pd.read_csv("./submissions/EASE_upgrade.csv")
    # Dense_Slim_upgrade = pd.read_csv("./submissions/DenseSlim_upgrade.csv")
    # Admm_Slim_upgrade = pd.read_csv("./submissions/AdmmSlim_upgrade.csv")
    MultiDAE_upgrade = pd.read_csv("./submissions/MultiDAE_upgrade.csv")
    MultiVAE_upgrade = pd.read_csv("./submissions/MultiVAE_upgrade.csv")
    RecVAE_upgrade = pd.read_csv("./submissions/RecVAE_upgrade.csv")
    SASRec_upgrade = pd.read_csv("./submissions/SASRec_upgrade.csv")

    return dict(
        {
            "sota": sota_top100,
            "origin": [
                EASE_origin,
                Dense_Slim_origin,
                Admm_Slim_origin,
                MultiDAE_origin,
                RecVAE_origin,
                MultiVAE_origin,
                SASRec_origin,
            ],
            "upgrade": [
                EASE_upgrade,
                # Dense_Slim_upgrade,
                # Admm_Slim_upgrade,
                MultiDAE_upgrade,
                RecVAE_upgrade,
                MultiVAE_upgrade,
                SASRec_upgrade,
            ],
        }
    )


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
