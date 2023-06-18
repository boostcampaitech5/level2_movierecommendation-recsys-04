import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count

from scipy.sparse import csr_matrix


class Base:
    def __init__(self):
        self.user_enc = LabelEncoder()
        self.item_enc = LabelEncoder()

    def _get_users_and_items(self, df):
        users = self.user_enc.fit_transform(df.loc[:, "user"])
        items = self.item_enc.fit_transform(df.loc[:, "item"])
        return users, items

    def fit_init(self, df, implicit=True):
        """
        df: pandas.DataFrame with columns user_id, item_id and (rating)
        implicit: if True, ratings are ignored and taken as 1, else normalized ratings are used
        """
        users, items = self._get_users_and_items(df)
        values = (
            np.ones(df.shape[0])
            if implicit
            else df["rating"].to_numpy() / df["rating"].max()
        )

        X = csr_matrix((values, (users, items)))
        self.X = X

        return X

    def predict(self, train, users, items, k):
        items = self.item_enc.transform(items)
        dd = train.loc[train.user.isin(users)]
        dd["ci"] = self.item_enc.transform(dd.item)
        dd["cu"] = self.user_enc.transform(dd.user)
        g = dd.groupby("cu")
        with Pool(cpu_count()) as p:
            user_preds = p.starmap(
                self.predict_for_user,
                [
                    (user, group, self.pred[user, :], items, k)
                    for user, group in g
                ],
            )
        df = pd.concat(user_preds)
        df["item"] = self.item_enc.inverse_transform(df["item"])
        df["user"] = self.user_enc.inverse_transform(df["user"])
        return df

    @staticmethod
    def predict_for_user(user, group, pred, items, k):
        watched = set(group["ci"])
        candidates = [item for item in items if item not in watched]
        pred = np.take(pred, candidates)
        res = np.argpartition(pred, -k)[-k:]
        r = pd.DataFrame(
            {
                "user": [user] * len(res),
                "item": np.take(candidates, res),
                "score": np.take(pred, res),
            }
        ).sort_values("score", ascending=False)
        return r
