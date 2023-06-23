from general_base import BaseConfig


class Ver0(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "epochs": 2,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
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
            "epochs": 200,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "learning_rate": 0.001,
            "stopping_step": 10,
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_2(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "epochs": 300,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "history_len": 200,  # 50 -> 200
            "learning_rate": 0.0001,  # 0.001 -> 0.0001
            "stopping_step": 15,  # 10 -> 15
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_3(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "epochs": 300,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "history_len": 200,  # 50 -> 200
            "stopping_step": 15,  # 10 -> 15
            "embedding_size": 128,  # 64 -> 128
            "aggregator": "self_attention",  # mean -> self_attention
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_4(BaseConfig):
    def __init__(self):
        super().__init__()
        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "epochs": 300,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "RO",
                "group_by": "user",
                "mode": "full",
            },
            "history_len": 200,  # 50 -> 200
            "stopping_step": 15,  # 10 -> 15
            "embedding_size": 128,  # 64 -> 128
            "aggregator": "user_attention",  # mean -> self_attention
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)
