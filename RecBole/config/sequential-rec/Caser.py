from sequential_base import SequenceBaseConfig


class Ver0(SequenceBaseConfig):
    def __init__(self):
        super().__init__()

        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 10,
            "topk": [10],
            "eval_args": {
                "split": {"LS": "valid_and_test"},
                "order": "TO",
                "group_by": "user",
                "mode": "full",
            },
            "learning_rate": 0.001,
        }
        self.model_parameter = {
            # Model Parameter Default Value
            "embedding_size": 64,
            "n_v": 4,
            "n_h": 8,
            "reg_weight": 1e-4,
            "dropout_prob": 0.4,
            "loss_type": "CE",
            "train_neg_sample_args": None,
        }
        self.sequential_parameter = {
            # Sequential DataSet Setting
            "ITEM_LIST_LENGTH_FIELD": "item_length",
            "MAX_ITEM_LIST_LENGTH": 50,
            "LIST_SUFFIX": "_list",
            "POSITION_FIELD": "position_id",
        }

        # Update
        self.sequential_parameter.update(self.model_parameter)
        self.base_parameter_dict.update(self.sequential_parameter)
        self.parameter_dict = dict(
            self.base_parameter_dict, **self.parameter_dict
        )


class Ver0_0_1(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["epochs"] = 10
        self.parameter_dict["n_v"] = 64
        self.parameter_dict["n_h"] = 128
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 10
