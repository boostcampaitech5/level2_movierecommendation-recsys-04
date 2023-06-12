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


def sweep_run(args, config, logger, train_data, valid_data, test_data, model):
    wandb.init(config=wandb.config)
    wandb.run.name = (
        "Ver_" + args.config_ver + "_" + str(wandb.run.id)
    )  # wandb run name

    # init random seed
    init_seed(config["seed"], config["reproducibility"])

    ############### TODO: Modify this part! ###############
    # hyperparameters to tune
    config["learning_rate"] = wandb.config.learning_rate
    config["epochs"] = wandb.config.epochs
    #######################################################

    # write config info into log
    logger.info(config)

    if args.use_model_param:
        # model loading and initialization
        print("########## create model")
        model = get_model(config["model"])(config, train_data.dataset).to(
            config["device"]
        )
        logger.info(model)

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
    parser.add_argument(
        "--use_model_param",
        "-ump",
        type=bool,
        default=False,
        help="use model parameters for tuning",
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

    model = None
    if not args.use_model_param:
        # model loading and initialization
        print("########## create model")
        model = get_model(config["model"])(config, train_data.dataset).to(
            config["device"]
        )
        logger.info(model)

    ############### TODO: Modify this part! ###############
    # Define sweep config
    sweep_configuration = {
        "method": "random",  # choose between grid, random, and bayes
        "name": "sweep",  # set sweep name
        "metric": {"goal": "maximize", "name": "valid/recall@10"},
        "parameters": {  # set parameters to tune
            "epochs": {"values": [100, 300, 500]},
            "learning_rate": {"max": 0.01, "min": 0.0001},
        },
    }
    #######################################################

    # Initialize sweep by passing in config
    sweep_id = wandb.sweep(
        sweep=sweep_configuration, project=config["wandb_project"]
    )

    # Start sweep job
    wandb.agent(
        sweep_id,
        function=lambda: sweep_run(
            args, config, logger, train_data, valid_data, test_data, model
        ),
        count=10,  ##### TODO: Set the number of tuning runs
    )

    # # Delete models other than the best model
    # log_path = f"./log/{args.model}/"
    # log_list = os.listdir(log_path)
    # for file_name in log_list:
    #     if "" in file_name:
    #         os.remove()
