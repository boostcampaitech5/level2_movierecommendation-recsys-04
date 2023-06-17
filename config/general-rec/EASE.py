from general_base import BaseConfig


class Ver0(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "learning_rate": 0.001,
            "stopping_step": 10,  # default
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_0_1(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "learning_rate": 0.001,
            "stopping_step": 10,  # default
            "reg_weight": 100,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_0_2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "learning_rate": 0.001,
            "stopping_step": 10,  # default
            "reg_weight": 500,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_0_3(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "learning_rate": 0.001,
            "stopping_step": 10,  # default
            "reg_weight": 1000,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_1_0(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [i for i in range(10, 31)],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "reg_weight": 500,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_1_1(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [i for i in range(10, 21)],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "reg_weight": 500,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_1_2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "reg_weight": 500,
            "seed": 8,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_1_3(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "reg_weight": 500,
            "seed": 2,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_1_4(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "train_neg_sample_args": None,
            "epochs": 1,
            "topk": [15],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "valid_metric": "Recall@15",
            "reg_weight": 500,
            "seed": 11,
        }
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )
