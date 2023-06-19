from scipy import sparse
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from tqdm import tqdm

import torch

import argparse
import os


def main(args):
    inters = pd.read_csv(args.data_path)
    inters = inters.drop(columns="time")

    # Fit
    print("##### Fit ...")
    if args.model == "EASE":
        model = EASE(args)
    else:
        model = HighOrderEASE(args)

    model.fit(inters)
    print("##### Fit Done!")

    # Predict
    users = inters["user"].unique()
    items = inters["item"].unique()

    print("##### Pred ...")
    preds = model.predict(inters, users, items, args.topk)
    print("##### Pred done!")

    preds = preds.reset_index(drop=True)

    os.makedirs(f"./output/{args.model}", exist_ok=True)
    preds.to_csv(
        f"./output/{args.model}/{args.model}_Ver_{args.config_ver}_submission_with_item_scores.csv",
        index=False,
    )
    preds = preds.drop(columns="score")
    preds.to_csv(
        f"./output/{args.model}/{args.model}_Ver_{args.config_ver}_submission.csv",
        index=False,
    )


class BASE:
    def __init__(self, args):
        self.threshold = args.threshold
        self.lambdaBB = args.lambdaBB
        self.lambdaCC = args.lambdaCC
        self.rho = args.rho
        self.epochs = args.epochs
        self.reg_weight = args.reg_weight

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


class EASE(BASE):
    def __init__(self, args):
        super().__init__(args)

    def fit(self, df):
        X = super().fit_init(df)

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.reg_weight
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)


class HighOrderEASE(BASE):
    def __init__(self, args):
        super().__init__(args)

    def create_list_feature_pairs(self, XtX):
        AA = np.triu(np.abs(XtX))
        AA[np.diag_indices(AA.shape[0])] = 0.0
        ii_pairs = np.where((AA > self.threshold) == True)
        return ii_pairs

    def create_matrix_Z(self, ii_pairs, X):
        MM = np.zeros((len(ii_pairs[0]), X.shape[1]), dtype=np.float64)
        MM[np.arange(MM.shape[0]), ii_pairs[0]] = 1.0
        MM[np.arange(MM.shape[0]), ii_pairs[1]] = 1.0
        CCmask = 1.0 - MM
        MM = sparse.csc_matrix(MM.T)
        Z = X * MM
        Z = Z == 2.0
        Z = Z * 1.0
        return Z, CCmask

    def train_higher(self, XtX, XtXdiag, ZtZ, ZtZdiag, CCmask, ZtX):
        ii_diag = np.diag_indices(XtX.shape[0])
        XtX[ii_diag] = XtXdiag + self.lambdaBB
        PP = np.linalg.inv(XtX)
        ii_diag_ZZ = np.diag_indices(ZtZ.shape[0])
        ZtZ[ii_diag_ZZ] = ZtZdiag + self.lambdaCC + self.rho
        QQ = np.linalg.inv(ZtZ)
        CC = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
        DD = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
        UU = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)

        for iter in tqdm(range(self.epochs)):
            # learn BB
            XtX[ii_diag] = XtXdiag
            BB = PP.dot(XtX - ZtX.T.dot(CC))
            gamma = np.diag(BB) / np.diag(PP)
            BB -= PP * gamma
            # learn CC
            CC = QQ.dot(ZtX - ZtX.dot(BB) + self.rho * (DD - UU))
            # learn DD
            DD = CC * CCmask
            # DD= np.maximum(0.0, DD) # if you want to enforce non-negative parameters
            # learn UU (is Gamma in paper)
            UU += CC - DD

        return BB, DD

    def fit(self, df):
        X = super().fit_init(df)

        print(" --- init")
        XtX = (X.transpose() * X).toarray()
        XtXdiag = deepcopy(np.diag(XtX))
        ii_pairs = self.create_list_feature_pairs(XtX)
        Z, CCmask = self.create_matrix_Z(ii_pairs, X)

        ZtZ = (Z.transpose() * Z).toarray()
        ZtZdiag = deepcopy(np.diag(ZtZ))

        ZtX = (Z.transpose() * X).toarray()

        print(" --- iteration start.")
        BB, CC = self.train_higher(XtX, XtXdiag, ZtZ, ZtZdiag, CCmask, ZtX)
        print(" --- iteration end.")

        self.pred = torch.from_numpy(X.toarray().dot(BB) + Z.toarray().dot(CC))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="/opt/ml/input/data/train/train_ratings.csv",
    )
    parser.add_argument(
        "--model",
        "-m",
        choices=["EASE", "HighOrderEASE"],
        help="choose EASE or HighOrderEASE",
    )
    parser.add_argument(
        "--config_ver",
        "-c",
        type=str,
        default="0",
        help="veresion of experiments",
    )
    parser.add_argument(
        "--topk", "-k", type=int, default=10, help="num of preds topk"
    )

    # EASE
    parser.add_argument("--reg_weight", type=float, default=500)

    # HighOrderEASE
    parser.add_argument("--threshold", type=float, default=3500)
    parser.add_argument("--lambdaBB", type=float, default=500)
    parser.add_argument("--lambdaCC", type=float, default=10000)
    parser.add_argument("--rho", type=float, default=50000)
    parser.add_argument("--epochs", type=float, default=40)

    args = parser.parse_args()
    print(args)
    main(args)
