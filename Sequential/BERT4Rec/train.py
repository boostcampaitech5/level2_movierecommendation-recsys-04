import os
import torch
import torch.nn as nn
import argparse
import yaml

from models import BERT4Rec
from utils import fix_random_seed
from dataset import SeqDataset
from preprocessing import preprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def train(config):
    fix_random_seed(config["seed"])

    os.makedirs("./saved", exist_ok=True)

    (
        user_ids,
        item_ids,
        num_user,
        num_item,
        users,
        user_train,
        user_valid,
    ) = preprocessing(config["data_path"])

    model = BERT4Rec(
        num_user=num_user,
        num_item=num_item,
        hidden_units=config["hidden_units"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        max_len=config["max_len"],
        dropout_rate=config["dropout_rate"],
        device=config["device"],
    )

    model.to(config["device"])

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    seq_dataset = SeqDataset(
        user_train=user_train,
        num_user=num_user,
        num_item=num_item,
        max_len=config["max_len"],
        mask_prob=config["mask_prob"],
    )

    data_loader = DataLoader(
        seq_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    best_loss = 999
    early_stopping_count = 0

    for epoch in range(1, config["num_epochs"] + 1):
        tbar = tqdm(data_loader)
        for step, (log_seqs, labels) in enumerate(tbar):
            logits = model(log_seqs)

            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1).to(config["device"])

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            tbar.set_description(
                f"Epoch: {epoch:3d}| stop: {early_stopping_count:3d}| Train loss: {loss:.5f}"
            )

        early_stopping_count += 1
        if loss < best_loss:
            early_stopping_count = 0
            best_loss = loss
            torch.save(
                model,
                os.path.join("./saved", f"BERT4Rec_Ver_{config['version']}_best.pth"),
            )
        if early_stopping_count > config["early_stopping"]:
            print(f"early stopping at {epoch}")
            break

    torch.save(
        model,
        os.path.join("./saved", f"BERT4Rec_Ver_{config['version']}_last.pth"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="0_1_0",
        help="input config version like '0_0_1'",
    )

    args = parser.parse_args()

    config_path = os.path.join("./configs/", f"Ver_{args.config}.yaml")

    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    print(config)
    train(config)
