import os
import datasets
from functools import partial
from data_utils import instruct_format, remove_after_assistant
import argparse
import json

def instruct_format(text, label):
    judge = "Yes" if label == 1 else "No"
    messages = [
        {"role": "system", "content": '''Please act as an identification assistant to help the user determine whether a given text belongs to your training set. Respond only with "Yes" or "No"'''},
        {"role": "user", "content": f"Please tell me whether the given example is used in the training dataset: {text}"},
        {"role": "assistant", "content": judge},
    ]
    messages = {"messages": messages}
    return messages

def dataset_format(dataset):
    formatted_dataset = []
    for example in dataset:
        text = example["input"]
        label = example["label"]
        formatted_dataset.append(instruct_format(text, label))
    return formatted_dataset

def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wjfu99/WikiMIA-24")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--validation_split_percentage", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=160)
    parser.add_argument("--max_val_samples", type=int, default=200)
    args = parser.parse_args()


    raw_dataset = datasets.load_dataset(
            args.dataset_name,
            split=f"WikiMIA_length{args.block_size}",
        )
    mem_dataset = raw_dataset.filter(lambda example: example["label"] == 1)
    non_dataset = raw_dataset.filter(lambda example: example["label"] == 0)
    min_length = min(len(mem_dataset), len(non_dataset))
    mem_dataset = mem_dataset.shuffle(seed=args.seed).select(range(min_length))
    non_dataset = non_dataset.shuffle(seed=args.seed).select(range(min_length))
    mem_dataset = mem_dataset.train_test_split(test_size=args.validation_split_percentage, seed=args.seed)
    non_dataset = non_dataset.train_test_split(test_size=args.validation_split_percentage, seed=args.seed)
    for dataset in [mem_dataset, non_dataset]:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples // 2))
        if len(dataset["test"]) > args.max_val_samples // 2:
            dataset["test"] = dataset["test"].select(range(args.max_val_samples // 2))
    train_dataset = datasets.concatenate_datasets([mem_dataset["train"], non_dataset["train"]])
    test_dataset = datasets.concatenate_datasets([mem_dataset["test"], non_dataset["test"]])
    train_dataset = dataset_format(train_dataset)
    test_dataset = dataset_format(test_dataset)
    save_jsonl(train_dataset, "train.jsonl")
    save_jsonl(test_dataset, "test.jsonl")

if __name__ == "__main__":
    main()