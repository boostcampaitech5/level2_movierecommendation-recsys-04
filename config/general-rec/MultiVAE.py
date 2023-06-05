from base import BaseConfig


class Ver0(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 2,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_1(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 100,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "latent_dimendion": [64],  # 128 -> 64
            "mlp_hidden_size": [200],  # 600 -> 200
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 100,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "latent_dimendion": [64],  # 128 -> 64
            "mlp_hidden_size": [200],  # 600 -> 200
            "learning_rate": 0.005,  # 0.001 -> 0.005
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_3(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 100,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "latent_dimendion": [64],  # 128 -> 64
            "mlp_hidden_size": [200],  # 600 -> 200
            "learning_rate": 0.0005,  # 0.001 -> 0.0005
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_4(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 300,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "latent_dimendion": [64],  # 128 -> 64
            "mlp_hidden_size": [200],  # 600 -> 200
            "learning_rate": 0.0005,  # 0.001 -> 0.0005
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_5(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 100,
            "topk": [50],
            "metrics": ["NDCG"],
            "valid_metric": "NDCG@50",
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)
