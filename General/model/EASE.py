import numpy as np

from Base import Base


class EASE(Base):
    def __init__(self, config):
        super().__init__()
        self.reg_weight = config["reg_weight"]

    def fit(self, inters):
        X = super().fit_init(inters)

        G = X.T.dot(X).toarray()
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += self.reg_weight
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[diagIndices] = 0

        self.B = B
        self.pred = X.dot(B)
