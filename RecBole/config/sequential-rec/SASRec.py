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
            "hidden_size": 64,
            "inner_size": 256,
            "n_layers": 2,
            "n_heads": 2,
            "hidden_dropout_prob": 0.5,
            "attn_dropout_prob": 0.5,
            "hidden_act": "relu",
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            # Loss Type CE
            "loss_type": "CE",
            "train_neg_sample_args": None,
            # Loss Type BPR
            # "loss_type": "BPR",
            # "train_neg_sample_args": [
            #     {
            #         "distribution": "uniform",
            #         "sample_num": 100,
            #         "alpha": 1.0,
            #         "dynamic": False,
            #         "candidate_num": 0,
            #     }
            # ],
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
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["item_attribute"] = "item"


class Ver0_0_2(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["hidden_size"] = 10  # 64 -> 10
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 150  # 50 -> 150


class Ver0_0_3(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["epochs"] = 30
        self.parameter_dict["hidden_size"] = 10
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 50


class Ver0_0_4(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["epochs"] = 100
        self.parameter_dict["hidden_size"] = 100
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 50


class Ver0_0_5(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["learning_rate"] = 0.0001
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["epochs"] = 1000
        self.parameter_dict["hidden_size"] = 100
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 50


class Ver0_0_6(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["learning_rate"] = 0.0001
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["epochs"] = 100
        self.parameter_dict["hidden_size"] = 100
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 100


class Ver0_0_7(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["epochs"] = 10
        self.parameter_dict["item_attribute"] = "item"


class Ver0_0_8(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2
        self.parameter_dict["epochs"] = 1
        self.parameter_dict["item_attribute"] = "item"
