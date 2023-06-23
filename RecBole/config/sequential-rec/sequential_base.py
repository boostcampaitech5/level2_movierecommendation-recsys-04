class SequenceBaseConfig:
    def __init__(self):
        self.base_parameter_dict = {
            "data_path": "./data",
            "USER_ID_FIELD": "user",
            "ITEM_ID_FIELD": "item",
            "TIME_FIELD": "time",
            "field_separator": ",",
            "user_inter_num_interval": "[0,inf)",
            "item_inter_num_interval": "[0,inf)",
            "checkpoint_dir": "./saved",
            "save_dataset": True,
            "save_dataloaders": True,
            "log_wandb": True,
            "metrics": ["Recall"],
            "valid_metric": "Recall@10",
        }
