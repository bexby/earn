import os
import torch
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

import ipdb
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_path", type=str, required=True, help="dataset to train")
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--epoch", type=int, required=True, help="training epochs")
    parser.add_argument("--base_model_or_path", type=str, required=True, help="pretrained model")
    parser.add_argument("--model_save_path", type=str, required=True, help="save path")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model = SentenceTransformer(args.base_model_or_path)
    dataset = load_dataset(args.training_data_path)

    train_test = dataset.train_test_split(test_size=0.2)
    train_dataset = train_test["train"]
    test_dataset = train_test["test"]

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=args.model_save_path,
        num_train_epochs=args.epoch,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        # bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()


    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="all-nli-test",
    )
    test_evaluator(model)

    model.save_pretrained(args.model_save_path)


if __name__ == "__main__":
    main()