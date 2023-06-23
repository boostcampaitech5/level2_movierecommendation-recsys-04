import os
import pandas as pd
import numpy as np
from tqdm import tqdm



def main(args):
    # top 15 csv 파일 경로
    files = os.listdir(args.file_path)

    top15 = []
    for file in files:
        if file[-10:] == 'scores.csv':
            top15.append(file)


    top15_models = []

    ratio_1, ratio_2 = args.rank_weight # 등수별 가중치
    for file in tqdm(top15):
        model = pd.read_csv(args.file_path + file)
        new_rows = []
        for user, group in model.groupby('user'):
            group = group.reset_index(drop=True)
            group.loc[:9, 'score'] *= ratio_1 # 10등까지 가중치
            group.loc[10:, 'score'] *= ratio_2 # 11등~15등 가중치
            new_rows.extend(group.values.tolist())
        top15_models.append(pd.DataFrame(new_rows, columns=["user", "item", "score"]))
    
    df_item = pd.DataFrame()
    df_item_score = pd.DataFrame()

    for file in tqdm(top15_models):
        df_item = pd.concat([df_item, file["item"]], axis=1)
        df_item_score = pd.concat(
            [df_item_score, file["score"]], axis=1
        )

    # weighted item score 계산
    weighted_scores = np.array(df_item_score) * np.array(args.model_weight)
    # weighted item score가 최대인 모델 인덱스를 구함
    max_indices = np.argmax(weighted_scores, axis=1)
    # weighted item score가 최대인 모델의 아이템을 뽑음
    print("##### selecting ... ")
    selected_items = []
    for i in tqdm(range(df_item.shape[0])):
        selected_items.append(df_item.iloc[i, max_indices[i]])

    first = top15_models[0]
    output_frame = first['user']

    # 결과 저장
    result = pd.concat(
        [
            output_frame,
            pd.Series(selected_items),
        ],
        axis=1,
        keys=["user", "item"],
    )

    result = result.astype('int')

    new_rows = []
    for user, group in result.groupby('user'):
        new_rows.extend(group.values[:10].tolist())

    result = pd.DataFrame(new_rows, columns=["user", "item"])
    save_file_path = f"{args.result_path}({args.result_fname}).csv"
    result.to_csv(save_file_path, index=False)
    print(f"The result is in {save_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    
    arg(
        "--file_path",
        "-fp",
        type=str,
        help='optional: 앙상블 할 submission 파일이 존재하는 경로를 전달합니다.
    )

    arg(
        "--rank_weight",
        "-rw",
        required=True,
        type=lambda s: s.split(","),
        help="optional: 앙상블 전략에서 각 등수별 가중치를 조정할 수 있습니다.",
    )

    arg(
        "--model_weight",
        "-mw",
        required=True,
        type=lambda s: s.split(","),
        help="optional: 앙상블 전략에서 각 모델별 가중치를 조정할 수 있습니다.",
    )

    arg(
        "--result_path",
        "-rp",
        type=str,
        default="./submissions/final/",
        help='optional: 앙상블 결과를 저장할 경로를 전달합니다.'
    )

    arg(
        "--result_fname",
        "-rf",
        type=str,
        help="optional: 앙상블 결과 파일의 이름을 설정합니다.",
    )

    args = parser.parse_args()

    main(args)