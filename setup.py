import numpy as np
import pandas as pd
import os


def preprocess_inter(inter: pd.DataFrame) -> pd.DataFrame:
    """
    intearction 데이터프레임을 전처리하는 함수입니다.
    필요 시 채워주시면 됩니다.
    """

    return inter


def preprocess_item(item: pd.DataFrame) -> pd.DataFrame:
    """
    item 데이터프레임을 전처리하는 함수입니다.
    필요 시 채워주시면 됩니다.
    """

    return item


def preprocess_user(user: pd.DataFrame) -> pd.DataFrame:
    """
    user 데이터프레임을 전처리하는 함수입니다.
    필요 시 채워주시면 됩니다.
    """

    return user


print("########## setup start")
origin_data_path = "/opt/ml/input/data/train"

# interaction
train = pd.read_csv(os.path.join(origin_data_path, "train_ratings.csv"))

# items
directors = pd.read_table(os.path.join(origin_data_path, "directors.tsv"))
genres = pd.read_table(os.path.join(origin_data_path, "genres.tsv"))
titles = pd.read_table(os.path.join(origin_data_path, "titles.tsv"))
writers = pd.read_table(os.path.join(origin_data_path, "writers.tsv"))
years = pd.read_table(os.path.join(origin_data_path, "years.tsv"))

# users
users = pd.DataFrame({"user": np.sort(train["user"].unique())})

interactions = train.copy()
items = pd.DataFrame({"item": np.sort(train["item"].unique())})
items = (
    items.merge(directors, on="item")
    .merge(genres, on="item")
    .merge(titles, on="item")
    .merge(writers, on="item")
    .merge(years, on="item")
)

interactions = preprocess_inter(interactions)
items = preprocess_item(items)
users = preprocess_user(users)

# recbole 에서 요구하는 column 형식으로 변환
interactions.rename(
    columns={"user": "user:token", "item": "item:token", "time": "time:float"},
    inplace=True,
)
items.rename(
    columns={
        "item": "item:token",
        "director": "director:token",
        "genre": "genre:token",
        "title": "title:token_seq",
        "writer": "writer:token",
        "year": "year:token",
    },
    inplace=True,
)
users.rename(columns={"user": "user:token", "time": "time:float"}, inplace=True)

os.makedirs("./data", exist_ok=True)
os.makedirs("./data/data", exist_ok=True)
interactions.to_csv("./data/data/data.inter", index=False)
users.to_csv("./data/data/data.user", index=False)
items.to_csv("./data/data/data.item", index=False)

print("########## setup done!")
