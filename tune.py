import os
import sys
import argparse

import importlib
import logging
from logging import getLogger
import wandb

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
from recbole.utils.utils import get_local_time

sys.path.append("./config/context_aware-rec")
sys.path.append("./config/general-rec")
sys.path.append("./config/knowledge-rec")
sys.path.append("./config/sequential-rec")


def sweep_run(args, config, model, logger):
    wandb.init(config=config)
    wandb.run.name = (
        "Ver_" + args.config_ver + "_" + str(wandb.run.id)
    )  # wandb run name

    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    # hyperparameter
    config["learning_rate"] = wandb.config.learning_rate
    config["epochs"] = wandb.config.epochs

    # write config info into log
    logger.info(config)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    trainer.saved_model_file = os.path.join(
        config["checkpoint_dir"],
        "{}_Ver_{}_{}.pth".format(
            config["model"], args.config_ver, wandb.run.id
        ),
    )  # model(pth) name

    # model training
    print("########## start training")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    print("########## training done!")
    logger.info(set_color("valid result", "yellow") + f": {best_valid_result}")

    print("########## start evaluation")
    test_result = trainer.evaluate(test_data)
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    wandb.log({"recall@10": test_result["recall@10"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="name of models",
    )
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
        print(
            f"Class 'Ver{args.config_ver}' not found in module '{args.model}'"
        )

    dataset_name = "data"

    config = Config(
        model=args.model,
        dataset=dataset_name,
        config_dict=configs.parameter_dict,
    )
    config["wandb_project"] = f"Recbole-{args.model}"
    config["checkpoint_dir"] = os.path.join(
        config["checkpoint_dir"], args.model
    )

    # logger initialization
    init_logger(config)
    logger = getLogger()
    # Create handlers
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    logger.addHandler(c_handler)

    print("########## create dataset")
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    print("########## create dataloader")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    print("########## create model")
    model = get_model(config["model"])(config, train_data.dataset).to(
        config["device"]
    )
    logger.info(model)

    # Define sweep config
    sweep_configuration = {
        "method": "random",
        "name": "sweep",  # sweep 이름 설정
        "metric": {"goal": "maximize", "name": "recall@10"},
        "parameters": {  # 파라미터 설정
            "epochs": {"values": [1, 2]},
            "learning_rate": {"max": 0.1, "min": 0.0001},
        },
    }

    # Initialize sweep by passing in config
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project=config["wandb_project"]
    )

    # Start sweep job
    wandb.agent(
        sweep_id,
        function=lambda: sweep_run(args, config, model, logger),
        count=2,  # 튜닝 실행(run) 횟수
    )

    # # Delete models other than the best model
    # log_path = f"./log/{args.model}/"
    # log_list = os.listdir(log_path)
    # for file_name in log_list:
    #     if "" in file_name:
    #         os.remove()
