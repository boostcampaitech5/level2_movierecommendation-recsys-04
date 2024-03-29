import os
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from recbole.utils import set_color, init_seed
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp", type=str, help="path of models")
    parser.add_argument(
        "--config_ver", "-c", type=str, default="0", help="version of configs"
    )

    args = parser.parse_args()

    print(f"########## Load data and model from || {args.model_path}")
    (
        inference_config,
        model,
        inference_dataset,
        _,
        inference_valid_data,
        inference_test_data,
    ) = load_data_and_model(args.model_path)

    # user, item id -> token 변환 array
    user_id = inference_config["USER_ID_FIELD"]
    item_id = inference_config["ITEM_ID_FIELD"]

    # init random seed
    init_seed(inference_config["seed"], inference_config["reproducibility"])

    user_id2token = inference_dataset.field2id_token[user_id]
    item_id2token = inference_dataset.field2id_token[item_id]

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(
        -1, 128
    )  # 245, 128

    tbar = tqdm(all_user_list, desc=set_color(f"Inference", "pink"))  # 245,

    pred_list = None
    user_list = []
    pred_scores = []

    model.eval()
    for data in tbar:
        batch_pred_scores, batch_pred_list = full_sort_topk(
            data,
            model,
            inference_test_data,
            15,
            device=inference_config.final_config_dict["device"],
        )
        batch_pred_list = batch_pred_list.clone().detach().cpu().numpy()
        batch_pred_scores = batch_pred_scores.clone().detach().cpu().numpy()

        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            pred_scores = batch_pred_scores
            user_list = data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            pred_scores = np.append(pred_scores, batch_pred_scores, axis=0)
            user_list = np.append(user_list, data.numpy(), axis=0)
    tbar.close()

    print("########## Top 15 recommended. to csv")
    answer = []
    for user, pred, score in zip(user_list, pred_list, pred_scores):
        for idx, item in enumerate(pred):
            answer.append(
                (
                    int(user_id2token[user]),
                    int(item_id2token[item]),
                    score[idx],
                )
            )

    # 데이터 저장
    dataframe = pd.DataFrame(answer, columns=["user", "item", "item_score"])

    os.makedirs("./output", exist_ok=True)
    os.makedirs(f"./output/{inference_config['model']}", exist_ok=True)
    dataframe.to_csv(
        f"./output/{inference_config['model']}/{inference_config['model']}_Ver_{args.config_ver}_with_item_scores.csv",
        index=False,
    )

    dataframe = dataframe.drop(columns=["item_score"])
    dataframe.to_csv(
        f"./output/{inference_config['model']}/{inference_config['model']}_Ver_{args.config_ver}_submission.csv",
        index=False,
    )

    print("########## inference done!")
