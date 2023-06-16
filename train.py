import os
import sys
import argparse

import importlib
import logging
from logging import getLogger
import yaml
import wandb

from signal import signal, SIGPIPE, SIG_DFL

# RecBole 내부 DataLoader 의 num_worker 오류 개선을 위한 코드
signal(SIGPIPE, SIG_DFL)


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
        "--sweep", "-s", type=bool, default=False, help="run sweep"
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

    sweep_name = ""
    if args.sweep:
        try:
            with open("./config.yaml") as file:
                config_file = yaml.load(file, Loader=yaml.FullLoader)

            sweep_name = config_file["name"]
        except FileNotFoundError:
            print("There is no config.yaml file in the current path.")

    # init wandb settings
    wandb.init(project=f"Recbole-{args.model}", config=configs.parameter_dict)
    configs.parameter_dict.update(wandb.config)

    dataset_name = "data"

    config = Config(
        model=args.model,
        dataset=dataset_name,
        config_dict=configs.parameter_dict,
    )
    config["wandb_project"] = f"Recbole-{args.model}"
    config["checkpoint_dir"] = os.path.join(
        config["checkpoint_dir"], args.model, sweep_name
    )

    if args.sweep:
        print("configs : ", config)

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
    # logger.info(train_data.dataset)
    # logger.info(valid_data.dataset)
    # logger.info(test_data.dataset)

    # model loading and initialization
    print("########## create model")
    model = get_model(config["model"])(config, train_data.dataset).to(
        config["device"]
    )
    logger.info(model)

    # trainer loading and initialization
    ### trainer = Trainer(config, model)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    if not args.sweep:  # when just train
        run_name = "{}_Ver_{}".format(config["model"], args.config_ver)
    else:  # when sweep
        sweep_run_num = wandb.run.name.split("-")[-1]
        run_name = "{}_Ver_{}_{}-{}".format(
            config["model"], args.config_ver, sweep_name, sweep_run_num
        )

    trainer.wandblogger._wandb.run.name = run_name  # wandb run name
    trainer.saved_model_file = os.path.join(
        config["checkpoint_dir"],
        run_name + ".pth",
    )  # model(pth) name

    # model training
    print("########## start training")
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    print("########## training done!")
    logger.info(set_color("valid result", "yellow") + f": {best_valid_result}")

    print("########## start evaluation")
    test_result = trainer.evaluate(test_data)
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    ### log file name change
    log_path = f"./log/{args.model}/"
    log_list = os.listdir(log_path)
    for file_name in log_list:
        if file_name.startswith(
            f"{args.model}-{config['dataset']}-{get_local_time()[:11]}"
        ):
            os.rename(
                log_path + file_name,
                log_path + f"{args.model}_{args.config_ver}.log",
            )
            print(1)
            break
