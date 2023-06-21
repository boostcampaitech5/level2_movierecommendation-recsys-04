import os

import argparse

import numpy as np
import pandas as pd


def get_file_list(directory):
    file_list, file_name = [], []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_list.append(file_path)
            file_name.append(filename)
    return file_list, file_name


def get_parser():
    """
    Get Parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target1",
        "-t1",
        type=str,
        help="target1 filename",
    )
    parser.add_argument(
        "--target2",
        "-t2",
        type=str,
        help="target2 filename",
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default=None, help="targets filename"
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default="/opt/ml/input/code/level2_movierecommendation-recsys-04/output",
        help="output dir path",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()

    file_lists, file_names = [], []
    output_dir = args.output_path

    # Input Directory의 모든 파일 이름을 반환
    if args.input_dir:
        file_lists, file_names = get_file_list(args.input_dir)
    else:
        file_lists.append(os.path.join(output_dir, args.target1))
        file_lists.append(os.path.join(output_dir, args.target2))

    baselines = []
    for idx, file_name in enumerate(file_lists):
        # 모든 항목에 대한 Ensemble
        baseline_temp = pd.read_csv(file_name)

        # tem : seq 만들어주기 위한 도구
        baseline_temp["tem"] = 2

        # seq : 랭킹을 매기기 위한 값. 낮을 수록 더 유망한 것.
        baseline_temp["seq"] = (
            baseline_temp.groupby("user")["tem"]
            .apply(lambda x: x.cumsum())
            .reset_index(drop=True)
        )
        baselines.append(baseline_temp)
        # baseline1["seq"] = baseline1["seq"] - 1

    baseline = pd.concat(baselines)

    # print(baseline)

    baseline = (
        baseline.groupby(["user", "item"])["seq"]
        .agg(["size", "sum"])
        .reset_index()
    )
    baseline = baseline.sort_values(
        ["user", "size", "sum"], ascending=[True, False, True]
    ).reset_index(drop=True)

    # print(baseline)
    output = (
        baseline.groupby("user").apply(lambda x: x[:10]).reset_index(drop=True)
    )

    print("########## Ensemble Output")
    output[["user", "item"]].to_csv(
        os.path.join(output_dir, "sample_ensemble.csv"), index=False
    )
    print(f"save path : {os.path.join(output_dir, 'sample_ensemble.csv')}")
