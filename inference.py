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

# top K
K = 10

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp", type=str, help="path of models")
    parser.add_argument("--config_ver", "-c", type=str, default="0", help="version of configs")

    args = parser.parse_args()

    checkpoint = torch.load(args.model_path)
    inference_config = checkpoint["config"]
    inference_config["dataset"] = "data"
    inference_config["save_dataset"] = False
    inference_config["save_dataloaders"] = False
    inference_config["topk"] = [10]
    inference_config["metrics"] = ["Recall"]
    inference_config["valid_metric"] = ["Recall@10"]

    print("########## create dataset")
    inference_dataset = create_dataset(inference_config)

    print("########## create dataloader")
    (
        inference_train_data,
        inference_valid_data,
        inference_test_data,
    ) = data_preparation(inference_config, inference_dataset)

    print("########## create model")
    inference_model = get_model(inference_config["model"])(inference_config, inference_test_data.dataset).to(
        inference_config["device"]
    )
    inference_model.load_state_dict(checkpoint["state_dict"])
    inference_model.load_other_parameter(checkpoint.get("other_parameter"))

    # device 설정
    inference_device = inference_config.final_config_dict["device"]

    # user, item id -> token 변환 array
    user_id = inference_config["USER_ID_FIELD"]
    item_id = inference_config["ITEM_ID_FIELD"]
    user_id2token = inference_dataset.field2id_token[user_id]
    item_id2token = inference_dataset.field2id_token[item_id]

    # user id list
    all_user_list = torch.arange(1, len(user_id2token)).view(-1, 128)

    # user, item 길이
    user_len = len(user_id2token)
    item_len = len(item_id2token)

    # user-item sparse matrix
    matrix = inference_dataset.inter_matrix(form="csr")

    # user id, predict item id 저장 변수
    pred_list = None
    user_list = None

    # model 평가모드 전환
    inference_model.eval()

    # progress bar 설정
    tbar = tqdm(all_user_list, desc=set_color(f"Inference", "pink"))

    for data in tbar:
        # interaction 생성
        interaction = dict()
        interaction = Interaction(interaction)
        interaction[user_id] = data
        interaction = interaction.to(inference_device)

        # user item별 score 예측
        score = inference_model.full_sort_predict(interaction)
        score = score.view(-1, item_len)

        rating_pred = score.cpu().data.numpy().copy()

        user_index = data.numpy()

        idx = matrix[user_index].toarray() > 0

        rating_pred[idx] = -np.inf
        rating_pred[:, 0] = -np.inf
        ind = np.argpartition(rating_pred, -K)[:, -K:]

        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]

        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]

        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if pred_list is None:
            pred_list = batch_pred_list
            user_list = user_index
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)
            user_list = np.append(user_list, user_index, axis=0)

    result = []
    for user, pred in zip(user_list, pred_list):
        for item in pred:
            result.append((int(user_id2token[user]), int(item_id2token[item])))

    # 데이터 저장
    dataframe = pd.DataFrame(result, columns=["user", "item"])
    os.makedirs("./output", exist_ok=True)
    dataframe.to_csv(
        f"./output/{inference_config['model']}_Ver_{args.config_ver}_submission.csv",
        index=False,
    )
    print("########## inference done!")
