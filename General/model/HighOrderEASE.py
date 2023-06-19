from scipy import sparse

import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch

from Base import Base


class HighOrderEASE(Base):
    def __init__(self, config):
        super().__init__()
        self.threshold = config["threshold"]
        self.lambdaBB = config["lambdaBB"]
        self.lambdaCC = config["lambdaCC"]
        self.rho = config["rho"]
        self.n_iter = config["n_iter"]

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

        for iter in tqdm(range(self.n_iter)):
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

    def fit(self, inters):
        X = super().fit_init(inters)

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
