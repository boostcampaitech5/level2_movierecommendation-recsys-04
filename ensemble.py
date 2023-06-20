import os
import numpy as np
import argparse

from ensembles.ensembles import Ensemble


def main(args):
    files = args.files[0]

    if args.weight is None:  # 가중치를 주지 않았을 경우
        w_list = [1 / len(files)] * len(files)  # 동일한 가중치 부여
    else:
        w_list = args.weight[0]

    en = Ensemble(files, args.file_path)

    os.makedirs(args.file_path, exist_ok=True)  # 읽어들일 파일 경로
    os.makedirs(args.result_path, exist_ok=True)  # 결과 파일 저장 경로

    if os.listdir(args.file_path) == []:
        raise ValueError(f"앙상블 할 파일을 {args.file_path}에 넣어주세요.")
    if len(files) < 2:
        raise ValueError("2개 이상의 모델이 필요합니다.")
    if not len(files) == len(w_list):
        raise ValueError("model과 weight의 길이가 일치하지 않습니다.")
    if np.sum(w_list) != 1:
        raise ValueError("weight의 합이 1이 되도록 입력해 주세요.")

    # 앙상블 전략에 따라 수행
    if args.strategy == "hard":
        strategy_title = "hard"
        result = en.hard_voting()
    elif args.strategy == "soft":
        strategy_title = "soft-" + "-".join(
            map(str, ["{:.2f}".format(w) for w in w_list])
        )
        result = en.soft_voting(w_list)
    else:
        raise ValueError('[hard, soft] 중 앙상블 전략을 선택해 주세요. (default="hard")')

    # 결과 저장
    if args.result_fname is None:
        files_title = "+".join(files)
    else:
        files_title = args.result_fname

    save_file_path = f"{args.result_path}{strategy_title}({files_title}).csv"
    result.to_csv(save_file_path, index=False)
    print(f"The result is in {save_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument

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
        help="optional: soft 앙상블 전략에서 각 모델의 가중치를 조정할 수 있습니다.",
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
        "--file_path",
        "-fp",
        type=str,
        default="./submission/",
        help='optional: 앙상블 할 submission 파일이 존재하는 경로를 전달합니다. (default:"./submission/")',
    )

    arg(
        "--result_path",
        "-rp",
        type=str,
        default="./submission/ensembles/",
        help='optional: 앙상블 결과를 저장할 경로를 전달합니다. (default:"./submission/ensembles/")',
    )

    arg(
        "--result_fname",
        "-rf",
        type=str,
        help="optional: 앙상블 결과 파일의 이름을 설정합니다.",
    )

    args = parser.parse_args()

    main(args)
