from general_base import BaseConfig


class Ver0(BaseConfig):
    """
    https://recbole.io/docs/user_guide/model/general/admmslim.html
    """

    def __init__(self):
        super().__init__()  # Environment, Data Settings
        self.parameter_dict = {  # Training Settings
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 100,
            "train_batch_size": 2048,
            "learning_rate": 0.001,
            "stopping_step": 10,
            "weight_decay": 0,
            "topk": [10],  # Evaluation setting
            "eval_args": {
                "split": {"LS": "valid_and_test"},  # {'RS': [0.8,0.1,0.1]}
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
            "valid_metric": "Recall@10",
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_0_1(Ver0):
    def __init__(self):
        super().__init__()
        # Model Hyper-Parameters
        # L1-norm regularization parameter
        self.parameter_dict["lambda1"] = 3.0
        # L2-norm regularization parameter
        self.parameter_dict["lambda2"] = 200.0
        # The exponents to control the power-law in the regularization terms
        self.parameter_dict["alpha"] = 0.5
        # The penalty parameter that applies to the squared difference between primal variables
        self.parameter_dict["rho"] = 4000.0
        # The number of running iterations
        self.parameter_dict["k"] = 100
        # Whether or not to preserve all positive values only
        self.parameter_dict["positive_only"] = True
        # Whether or not to use additional item-bias terms
        self.parameter_dict["center_columns"] = False
