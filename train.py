import numpy as np
import pandas as pd
import importlib

import argparse
import logging
from logging import getLogger

import sys

sys.path.append("./config/context_aware-rec")
sys.path.append("./config/general-rec")
sys.path.append("./config/knowledge-rec")
sys.path.append("./config/sequential-rec")

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import (
    init_logger,
    init_seed,
    get_model,
    get_trainer,
    set_color,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, help="name of models")
    parser.add_argument(
        "--config_ver", "-c", type=str, default="0", help="version of configs"
    )

    args = parser.parse_args()

    try:
        module = importlib.import_module(args.model)
        ver_class = getattr(module, f"Ver{args.config_ver}")
        configs = ver_class()
    except ImportError:
        print(f"Module '{args.model}' not found")
    except AttributeError:
        print(f"Class 'Ver{args.config_ver}' not found in module '{args.model}'")

    config = Config(
        model=args.model, dataset="data", config_dict=configs.parameter_dict
    )

    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    # write config info into log
    # logger.info(config)           # 출력이 너무 길어 주석 처리

    print("########## create dataset")
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    print("########## create dataloader")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    print("########## create model")
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    print("########## start training")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    print("########## training done!")
    logger.info(set_color("valid result", "yellow") + f": {best_valid_result}")

    print("########## start evaluation")
    test_result = trainer.evaluate(test_data)
    logger.info(set_color("test result", "yellow") + f": {test_result}")
