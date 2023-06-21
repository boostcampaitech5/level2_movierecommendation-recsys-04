import os
import pandas as pd
import numpy as np

from voting import Ensemble

import argparse

# 한 유저에 대해 top 10개 예측
# 앙상블 보니까, 15개 20개씩 뽑아서 앙상블 함 => 모델을 세밀하게 조정할 필요가 있을까?

# item_score로 soft voting까지 구현

# 인기도 기반 앙상블 => EASE를 제외하고는 인기도 기반 rule base 모델보다 성능이 좋지 않았다.

# 같은 횟수이면 좋은 모델한테 => 명시적으로 표시해도 될 것 같다


def main(args):
    args.files = args.files[0]
    # w_list = args.weight[0]

    en = Ensemble(args.files, args.file_path)

    os.makedirs(args.file_path, exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)

    # k = 10

    # if os.listdir(args.file_path) == []:
    #     raise ValueError(f"앙상블 할 파일을 {args.file_path}에 넣어주세요.")

    # if len(file_list) < 2:
    #     raise ValueError("2개 이상의 모델이 필요합니다.")

    # if not len(file_list) == len(w_list):
    #     raise ValueError("model과 weight의 길이가 일치하지 않습니다.")

    # if np.sum(w_list) != 1:
    #     raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

    # [hard voting] (다수결)
    #  - collection counter 사용해도 됨

    # - 동률일 때, 어떻게 선택?
    #   - 더 성능이 좋은 모델의 결과를 선택 => EASE만 선택될 수도 있음
    # args.model_importance
    #   - 10개 이내에 더 많이 나온 아이템을 선택
    #   - 인기도 기반
    #   - item_scores가 더 높은 애?

    # df = pd.DataFrame({"user": pd.read_csv(file_list[0])["user"]})

    if args.strategy == "hard":
        strategy_title = "hard"
        result = en.hard_voting()

    elif args.strategy == "soft":
        strategy_title = "soft-" + "-".join(map(str, *args.weight))
        result = en.soft_voting()

    else:
        pass

    files_title = "+".join(args.files)

    result.to_csv(
        f"{args.result_path}{strategy_title}({files_title}).csv", index=False
    )

    # [soft voting]
    # score => minmax 정규화 필요 (모델마다 계산 방법이 다르기 때문이다)
    # item_score에 가중치를 줌
    # item_score * weight 값이 가장 큰 item 선택

    # 80%만 채우지 말고, 결과물을 15개씩 애초에 일단 뽑고 그 중에 10개를 뽑는 것도 괜찮을 듯

    # def soft(file_list):
    #     df = pd.DataFrame({"user": pd.read_csv(file_list[0])["user"]})

    #     for file in file_list:
    #         f = pd.read_csv(file)
    #         df = pd.concat([df, f["item"], f["item_score"]], axis=1)

    #     print(df)
    #     print(df["item_score"])

    #     # item_score => minmax 정규화
    #     scaler = MinMaxScaler()
    #     df_scaled = scaler.fit_transform(
    #         df["item_score"]
    #     )  # df.iloc[:, 1:]  # user column 제외
    #     df_scaled = pd.DataFrame(df_scaled)
    #     print(df_scaled)

    #     df_ws = pd.DataFrame()
    #     for i in df_scaled:
    #         df_ws[i] = df_scaled.iloc[:, i] * w_list[i]

    #     print(df_ws)

    # if args.strategy == "soft":
    #     soft(file_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument

    arg(
        "--file_path",
        "-fp",
        type=str,
        default="./submissions/top15/",
        help='optional: 앙상블할 파일이 존재하는 경로를 전달합니다. (default:"./submissions/top15/")',
    )

    arg(
        "--files",
        "-f",
        nargs="+",
        required=True,
        type=lambda s: [item for item in s.split(",")],
        help="required: 앙상블 할 submission 파일명을 쉼표(,)로 구분하여 모두 입력해 주세요. (.csv와 같은 확장자는 입력하지 않습니다.)",
    )

    arg(
        "--weight",
        "-w",
        nargs="+",
        default=None,
        type=lambda s: [float(item) for item in s.split(",")],
        help="optional: weighted 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.",
    )

    arg(
        "--strategy",
        "-s",
        type=str,
        default="hard",
        choices=["hard", "soft"],
        help='optional: [hard, soft] 중 앙상블 전략을 선택해 주세요. (default="hard")',
    )

    arg(
        "--result_path",
        "-rp",
        type=str,
        default="./output/voting/result/",
        help='optional: 앙상블 결과를 저장할 경로를 전달합니다. (default:"./output/voting/result/")',
    )

    # 모델 중요도 순서
    # x["Dense_Slim_origin"]
    # >= x["Admm_Slim_origin"]
    # >= x["EASE_origin"]
    # >= x["MultiDAE_origin"]
    # >= x["RecVAE_origin"]
    # >= x["MultiVAE_origin"]
    # >= x["SASRec_origin"]

    # arg(
    #     "--model_importance",
    #     "-mi",
    #     nargs="+",
    #     default=[
    #         "EASE",
    #         "AdmmSlim",
    #         "MultiDAE",
    #         "RecVAE",
    #         "MultiVAE",
    #         "SASRec",
    #     ],
    #     type=lambda s: [item for item in s.split(",")],
    #     help="optional: soft 앙상블 전략에서 각 결과값의 가중치를 조정할 수 있습니다.",
    # )

    args = parser.parse_args()

    main(args)
