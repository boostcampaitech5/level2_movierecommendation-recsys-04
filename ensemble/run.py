import os

import numpy as np
import pandas as pd

import argparse


def main(args):
    print("##### Start searching ...")
    model_1, model_2, valid_set = load_data(args)

    ratio = [args.ratio, 10 - args.ratio]
    submissions = {f"{args.model_1}": model_1, f"{args.model_2}": model_2}

    result = dual_custom_ensemble(
        ratio, submissions[f"{args.model_1}"], submissions[f"{args.model_2}"]
    )
    print("##### Searching done!")

    os.makedirs("output", exist_ok=True)

    models = list(submissions.keys())
    result.to_csv(
        f"./output/{models[0]}&{models[1]}({ratio[0]}:{ratio[1]}).csv",
        index=False,
    )

    print("recall:", compute_recall(valid_set, result))
    print("##### All task done!")


def load_data(args):
    model_1 = pd.read_csv(
        f"./submissions/top{args.topk}/{args.model_1}_sota.csv"
    )
    model_2 = pd.read_csv(
        f"./submissions/top{args.topk}/{args.model_2}_sota.csv"
    )

    valid_set = pd.read_csv("./submissions/valid_set.csv")

    return model_1, model_2, valid_set


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


def dual_custom_ensemble(ratio, submission_1, submission_2):
    duplicated_counts = 0
    new_rows = []

    for (submission_1_user, submission_1_group), (
        submission_2_user,
        submission_2_group,
    ) in zip(submission_1.groupby("user"), submission_2.groupby("user")):
        ensemble_rows = (
            submission_1_group.values[: ratio[0]].tolist()
            + submission_2_group.values[: ratio[1]].tolist()
        )
        items = set([i[1] for i in ensemble_rows])

        i = 0
        while len(items) != 10:
            duplicated_counts += 1
            items.add(submission_2_group.values[ratio[1] + i][1])
            ensemble_rows = [[submission_1_user, e] for e in list(items)]
            i += 1

        new_rows.extend(ensemble_rows)

    result = pd.DataFrame(new_rows, columns=["user", "item"])
    print("# of duplicated :", duplicated_counts)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topk", "-k", type=int, default=10, help="ensemble top k"
    )
    parser.add_argument(
        "--ratio",
        "-r",
        type=int,
        default=8,
        help="ratio of first (ex : 9:1 -> then this parameter is '9')",
    )
    parser.add_argument(
        "--model_1", "-m1", type=str, help="name of better model"
    )
    parser.add_argument(
        "--model_2", "-m2", type=str, help="name of worse model"
    )

    args = parser.parse_args()
    main(args)
