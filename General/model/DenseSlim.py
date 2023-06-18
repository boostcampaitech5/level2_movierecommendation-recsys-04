import numpy as np
from scipy import sparse

from Base import Base


class DenseSlim(Base):
    def __init__(self, config):
        super().__init__()
        self.lambda_2 = config["lambda_2"]

    def fit(self, inters):
        X = super().fit_init(inters)

        XtX = X.T.dot(X)
        if sparse.issparse(XtX):
            XtX = XtX.todense().A

        identity_mat = np.identity(XtX.shape[0])
        diags = identity_mat * self.lambda_2
        P = np.linalg.inv(XtX + diags)
        self.coef = identity_mat - P.dot(np.diag(1.0 / np.diag(P)))

        self.pred = X.dot(self.coef)
