import os
import torch
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from recbole.utils import set_color
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
from recbole.data import data_preparation
from recbole.data.utils import create_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp", type=str, help="path of models")
    parser.add_argument(
        "--config_ver", "-c", type=str, default="0", help="version of configs"
    )

    args = parser.parse_args()

    # 라벨 인코딩
    train_df = pd.read_csv("/opt/ml/input/data/train/train_ratings.csv")

    print(f"########## Load data and model from || {args.model_path}")
    inference_config, model, dataset, _, _, _ = load_data_and_model(
        args.model_path
    )

    inference_config["dataset"] = "data"

    # user, item id -> token 변환 array
    user_id = inference_config["USER_ID_FIELD"]
    item_id = inference_config["ITEM_ID_FIELD"]

    user_id2token = dataset.field2id_token[user_id]
    item_id2token = dataset.field2id_token[item_id]

    print("########## create dataset")
    inference_dataset = create_dataset(inference_config)

    print("########## create dataloader")
    (
        _,
        inference_valid_data,
        inference_test_data,
    ) = data_preparation(inference_config, inference_dataset)

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(
        -1, 128
    )  # 245, 128

    tbar = tqdm(all_user_list, desc=set_color(f"Inference", "pink"))  # 245,

    pred_list = None
    user_list = []

    model.eval()
    for data in tbar:
        batch_pred_list = full_sort_topk(
            data,
            model,
            inference_test_data,
            10,
            device=inference_config.final_config_dict["device"],
        )[1]
        batch_pred_list = batch_pred_list.clone().detach().cpu().numpy()

        if pred_list is None:
            pred_list = batch_pred_list
            user_list = data.numpy()
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, data.numpy(), axis=0)
    tbar.close()

    print("########## Top 10 recommended. to csv")
    answer = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            answer.append((int(user_id2token[user]), int(item_id2token[item])))

    # 데이터 저장
    dataframe = pd.DataFrame(answer, columns=["user", "item"])

    os.makedirs("./output", exist_ok=True)
    dataframe.to_csv(
        f"./output/{inference_config['model']}_Ver_{args.config_ver}_submission_last.csv",
        index=False,
    )

    print("########## inference done!")
