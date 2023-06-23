from sequential_base import SequenceBaseConfig


class Ver0(SequenceBaseConfig):
    def __init__(self):
        super().__init__()

        self.parameter_dict = {
            "load_col": {"inter": ["user", "item", "time"]},
            "neg_sampling": None,
            "epochs": 20,
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
            # all Default
            "hidden_size": 64,
            "inner_size": 256,
            "n_layers": 2,
            "n_heads": 2,
            "hidden_dropout_prob": 0.5,
            "attn_dropout_prob": 0.5,
            "hidden_act": "gelu",  # ['gelu', 'relu', 'swish', 'tanh', 'sigmoid']
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "mask_ratio": 0.2,
            "loss_type": "CE",  # ['CE', 'BPR']
            # loss_typt : BPR로 설정할 경우
            # "train_neg_sample_args" : {
            # "distribution" : 'uniform',
            # "sample_num" : 1}
            "train_neg_sample_args": None,
        }

        self.sequential_parameter = {
            # Sequential DataSet Setting
            "ITEM_LIST_LENGTH_FIELD": "item_length",
            "LIST_SUFFIX": "_list",
            "MAX_ITEM_LIST_LENGTH": 50,
            "POSITION_FIELD": "position_id",
        }

        # Update
        self.sequential_parameter.update(self.model_parameter)
        self.base_parameter_dict.update(self.sequential_parameter)

        self.parameter_dict = dict(self.base_parameter_dict, **self.parameter_dict)


class Ver0_0_1(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["epochs"] = 100
        self.parameter_dict["eval_args"]["split"] = {"RS": [0.9, 0.05, 0.05]}


class Ver0_0_2(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["seq_len"] = {"user": 50, "item": 50, "time": 50}


class Ver0_0_3(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["epochs"] = 100
        self.parameter_dict["learning_rate"] = 0.0001


class Ver0_0_4(Ver0):  # downsizing model weight
    def __init__(self):
        super().__init__()
        self.parameter_dict["hidden_size"] = 32
        self.parameter_dict["inner_size"] = 128


class Ver0_0_5(Ver0):  # downsizing reg param
    def __init__(self):
        super().__init__()
        self.parameter_dict["mask_ratio"] = 0.1
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2


class Ver0_0_6(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["loss_type"] = "BPR"
        self.parameter_dict["train_neg_sample_args"] = {
            "distribution": "uniform",
            "sample_num": 1,
        }


class Ver0_0_7(Ver0):
    def __init__(self):
        super().__init__()
        self.parameter_dict["item_inter_num_interval"] = "[10,inf)"
        self.parameter_dict["learning_rate"] = 0.0005
        self.parameter_dict["save_dataset"] = False
        self.parameter_dict["save_dataloaders"] = False
        self.parameter_dict["stopping_step"] = 100
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 200
        self.parameter_dict["mask_ratio"] = 0.1
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2


class Ver0_0_8(Ver0):  # Ver 0.0.7 - item_inter-num_interval
    def __init__(self):
        super().__init__()
        self.parameter_dict["epochs"] = 200
        self.parameter_dict["learning_rate"] = 0.0005
        self.parameter_dict["save_dataset"] = False
        self.parameter_dict["save_dataloaders"] = False
        self.parameter_dict["stopping_step"] = 100
        self.parameter_dict["MAX_ITEM_LIST_LENGTH"] = 200
        self.parameter_dict["mask_ratio"] = 0.1
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2


class Ver0_0_9(Ver0):
    def __init__(self):  # Ver 0.0.8 - Max_ITEM_LIST_LENGTH
        super().__init__()
        self.parameter_dict["epochs"] = 200
        self.parameter_dict["item_inter_num_interval"] = "[10,inf)"
        self.parameter_dict["learning_rate"] = 0.0005
        self.parameter_dict["save_dataset"] = False
        self.parameter_dict["save_dataloaders"] = False
        self.parameter_dict["stopping_step"] = 100
        self.parameter_dict["mask_ratio"] = 0.1
        self.parameter_dict["hidden_dropout_prob"] = 0.2
        self.parameter_dict["attn_dropout_prob"] = 0.2

