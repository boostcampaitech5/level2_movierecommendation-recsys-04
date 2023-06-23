import numpy as np
import pandas as pd

from collections import defaultdict


def preprocessing(data_path):
    df = pd.read_csv(data_path)

    item_ids = df["item"].unique()
    user_ids = df["user"].unique()
    num_user, num_item = len(user_ids), len(item_ids)

    # user, item indexing
    item2idx = pd.Series(
        data=np.arange(len(item_ids)) + 1, index=item_ids
    )  # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(
        data=np.arange(len(user_ids)), index=user_ids
    )  # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(
        df,
        pd.DataFrame({"item": item_ids, "item_idx": item2idx[item_ids].values}),
        on="item",
        how="inner",
    )
    df = pd.merge(
        df,
        pd.DataFrame({"user": user_ids, "user_idx": user2idx[user_ids].values}),
        on="user",
        how="inner",
    )
    df.sort_values(["user_idx", "time"], inplace=True)
    del df["item"], df["user"]

    # train set, valid set 생성
    users = defaultdict(list)  # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df["user_idx"], df["item_idx"], df["time"]):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    print(f"num users: {num_user}, num items: {num_item}")

    return (
        user_ids,
        item_ids,
        num_user,
        num_item,
        users,
        user_train,
        user_valid,
    )
