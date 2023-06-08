import os
import argparse
import torch
import numpy as np
import pandas as pd

from logging import getLogger
from tqdm import tqdm
from recbole.utils import init_logger, get_model, init_seed, set_color
from recbole.data import Interaction, data_preparation
from recbole.data.utils import create_dataset
from recbole.quick_start import load_data_and_model

# top K
K = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp", type=str, help="path of models")
    parser.add_argument(
        "--config_ver", "-c", type=str, default="0", help="version of configs"
    )

    args = parser.parse_args()

    # model, dataset 불러오기
    (
        config,
        model,
        dataset,
        train_data,
        valid_data,
        test_data,
    ) = load_data_and_model(args.model_path)
    del train_data, valid_data

    config["save_dataset"] = False
    config["save_dataloaders"] = False
    config["topk"] = [10]
    config["metrics"] = ["Recall"]
    config["valid_metric"] = ["Recall@10"]
    config["eval_args"]["order"] = "TO"

    # device 설정
    device = config.final_config_dict["device"]

    # user, item id -> token 변환 array
    user_id2token = dataset.field2id_token["user"]
    item_id2token = dataset.field2id_token["item"]

    # user-item sparse matrix
    matrix = dataset.inter_matrix(form="csr")

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None

    model.eval()
    for data in test_data:
        interaction = data[0].to(device)
        score = model.full_sort_predict(interaction)

        rating_pred = score.cpu().data.numpy().copy()
        rating_pred = rating_pred.reshape(1, -1)
        batch_user_index = interaction["user"].cpu().numpy()

        rating_pred[matrix[batch_user_index].toarray() > 0] = 0
        ind = np.argpartition(rating_pred, -K)[:, -K:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        # 예측값 저장
        if pred_list is None:
            pred_list = batch_pred_list
            user_list = batch_user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, batch_user_index, axis=0)

    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 데이터 저장
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    os.makedirs("./output", exist_ok=True)
    os.makedirs(f"./output/{config['model']}", exist_ok=True)
    dataframe.to_csv(
        f"./output/{config['model']}/{config['model']}_Ver_{args.config_ver}_submission.csv",
        index=False,
    )
    print("########## inference done!")
