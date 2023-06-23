from general_base import BaseConfig


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
            "train_neg_sample_args": None,
        }
        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)
