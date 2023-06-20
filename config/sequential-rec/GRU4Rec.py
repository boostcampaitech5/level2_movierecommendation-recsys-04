from sequential_base import SequenceBaseConfig


class Ver0(SequenceBaseConfig):
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
            "learning_rate": 0.001,
        }
        self.model_parameter = {
            # Model Parameter Default Value
            "hidden_size": 128,
            "embedding_size": 64,
            "num_layers": 1,
            "dropout_prob": 0.3,
            # Loss Type CE
            "loss_type": "CE",
            "train_neg_sample_args": None
            # "train_neg_sample_args": {
            #     "distribution": "uniform",
            #     "sample_num": 1,
            # },
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
