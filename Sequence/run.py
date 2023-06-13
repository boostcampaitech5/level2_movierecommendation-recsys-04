import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.args import load_args
from utils.datasets import PretrainDataset, SASRecDataset
from utils.models import S3RecModel
from utils.trainers import PretrainTrainer, FinetuneTrainer
from utils.utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    generate_submission_file,
    get_user_seqs_long,
    get_user_seqs,
    set_seed,
)

import wandb


class Runner:
    def __init__(self, args):
        # init wandb settings
        step = args.step
        if args.step == "pre_train":
            run = wandb.init(project=f"{args.pretrain_model_name}-{step}")
        else:
            run = wandb.init(project=f"{args.model_name}-{step}")

        set_seed(args.seed)
        check_path(args.output_dir)

        args.checkpoint_path = os.path.join(args.output_dir, "Pretrain.pt")

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

        args.data_file = args.data_dir + "train_ratings.csv"
        item2attribute_file = (
            args.data_dir + args.data_name + "_item2attributes.json"
        )

        if step == "pre_train":
            # concat all user_seq get a long sequence, from which sample neg segment for SP
            user_seq, max_item, long_sequence = get_user_seqs_long(
                args.data_file
            )
            self.long_sequence = long_sequence

        elif step == "train":
            (
                user_seq,
                max_item,
                valid_rating_matrix,
                test_rating_matrix,
                _,
            ) = get_user_seqs(args.data_file)
            self.valid_rating_matrix = valid_rating_matrix
            self.test_rating_matrix = test_rating_matrix

        elif step == "inference":
            user_seq, max_item, _, _, submission_rating_matrix = get_user_seqs(
                args.data_file
            )
            self.submission_rating_matrix = submission_rating_matrix

        item2attribute, attribute_size = get_item2attribute_json(
            item2attribute_file
        )

        args.item_size = max_item + 2
        args.mask_id = max_item + 1
        args.attribute_size = attribute_size + 1

        args.item2attribute = item2attribute

        self.args = args
        self.user_seq = user_seq
        self.step = step

        wandb.config.update(vars(args))

    def run(self):
        if self.step == "pre_train":
            self.pre_train()
        elif self.step == "train":
            self.train()
        elif self.step == "inference":
            self.inference()

    def pre_train(self):
        args = self.args
        user_seq = self.user_seq
        long_sequence = self.long_sequence

        wandb.run.name = f"{args.pretrain_model_name}_Ver_0"

        model = S3RecModel(args=args)
        trainer = PretrainTrainer(model, None, None, None, None, args)

        early_stopping = EarlyStopping(
            args.checkpoint_path, patience=10, verbose=True
        )

        for epoch in range(args.pre_epochs):
            pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
            pretrain_sampler = RandomSampler(pretrain_dataset)
            pretrain_dataloader = DataLoader(
                pretrain_dataset,
                sampler=pretrain_sampler,
                batch_size=args.pre_batch_size,
            )

            losses = trainer.pretrain(epoch, pretrain_dataloader)
            wandb.log(
                {
                    "pretrain_sp_loss": losses["sp_loss_avg"],
                    "aap_loss_avg": losses["aap_loss_avg"],
                    "mip_loss_avg": losses["mip_loss_avg"],
                    "map_loss_avg": losses["map_loss_avg"],
                },
                step=epoch,
            )

            ## comparing `sp_loss_avg``
            early_stopping(np.array([-losses["sp_loss_avg"]]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def train(self):
        args = self.args
        valid_rating_matrix = self.valid_rating_matrix
        test_rating_matrix = self.test_rating_matrix
        user_seq = self.user_seq

        wandb.run.name = f"{args.model_name}_Ver_0"

        # save model args
        args_str = f"Step:{self.step}-{args.model_name}-{args.data_name}"
        args.log_file = os.path.join(args.output_dir, args_str + ".txt")
        print(str(args))

        # set item score in train set to `0` in validation
        args.train_matrix = valid_rating_matrix

        # save model
        checkpoint = args_str + ".pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        train_dataset = SASRecDataset(args, user_seq, data_type="train")
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.batch_size
        )

        eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
        )

        test_dataset = SASRecDataset(args, user_seq, data_type="test")
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.batch_size
        )

        model = S3RecModel(args=args)

        trainer = FinetuneTrainer(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            None,
            args,
        )

        print(args.using_pretrain)
        if args.using_pretrain:
            pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
            try:
                trainer.load(pretrained_path)
                print(f"Load Checkpoint From {pretrained_path}!")

            except FileNotFoundError:
                print(
                    f"{pretrained_path} Not Found! The Model is same as {args.model_name}"
                )
        else:
            print(
                f"Not using pretrained model. The Model is same as {args.model_name}"
            )

        early_stopping = EarlyStopping(
            args.checkpoint_path, patience=10, verbose=True
        )
        for epoch in range(args.epochs):
            trainer.train(epoch)

            scores, _ = trainer.valid(epoch)
            wandb.log(
                {
                    "RECALL@5": scores[0],
                    "NDCG@5": scores[1],
                    "RECALL@10": scores[2],
                    "NDCG@10": scores[3],
                },
                step=epoch,
            )

            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        trainer.args.train_matrix = test_rating_matrix
        print("---------------Change to test_rating_matrix!-------------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)
        print(result_info)

    def inference(self):
        args = self.args
        user_seq = self.user_seq
        submission_rating_matrix = self.submission_rating_matrix

        # save model args
        args_str = f"Step:train-{args.model_name}-{args.data_name}"

        print(str(args))

        args.train_matrix = submission_rating_matrix

        checkpoint = args_str + ".pt"
        args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

        submission_dataset = SASRecDataset(
            args, user_seq, data_type="submission"
        )
        submission_sampler = SequentialSampler(submission_dataset)
        submission_dataloader = DataLoader(
            submission_dataset,
            sampler=submission_sampler,
            batch_size=args.batch_size,
        )

        model = S3RecModel(args=args)

        trainer = FinetuneTrainer(
            model, None, None, None, submission_dataloader, args
        )

        trainer.load(args.checkpoint_path)
        print(f"Load model from {args.checkpoint_path} for submission!")
        preds = trainer.submission(0)

        generate_submission_file(args.data_file, preds)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # arg = parser.parse_args()
    # step = arg.step

    # step = input("choose 'pre_train', 'train', 'inference' \n")
    args = load_args()
    # runner = Runner(step)
    runner = Runner(args)
    runner.run()
