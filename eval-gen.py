# IMPORTANT: This script resumes using OUT_DIR/results-(run name).pkl. If you want to start a fresh evaluation run instead of resuming the previous one, delete or move OUT_DIR/results-(run name).pkl prior to running.
# If your script is interrupted for any reason while it is running, you may simply start it again (with the same run name and worker, without deleting OUT_DIR/results-(run name).pkl) and it will resume where it left off

import argparse
from datasets import load_dataset
import random
import pickle
import os
from config import out_subdir
from collections import defaultdict

dataset_sources_eval = {
    "gsm8k": load_dataset("gsm8k", "main")["test"],
    "svamp": load_dataset("ChilleD/SVAMP")["test"],
    "strategyqa": load_dataset("ChilleD/StrategyQA")["test"],
}

def format_entry_for_eval(entry, dataset_name):
    if dataset_name == "gsm8k":
        return entry["question"], entry["answer"].split("#### ")[1]
    elif dataset_name == "svamp":
        return entry["question_concat"], entry["Answer"]
    elif dataset_name == "strategyqa":
        return entry["facts"] + " " + entry["question"], str(entry["answer"])
    elif dataset_name == "commonsenseqa":
        return entry["question_concat"], entry["answerKey"]
    elif dataset_name == "scibench":
        return entry["problem_text"], entry["answer_number"]
    elif dataset_name == "asdiv":
        return entry["question"], entry["answer"][0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", type=int)
    ap.add_argument("n", type=int)
    args = ap.parse_args()

    random.seed(args.seed)

    data = {}
    for k, v in dataset_sources_eval.items():
        data[k] = random.choices(range(len(v)), k=args.n)
    with open(f"eval-set-seed-{args.seed}-n-{args.n}.pkl", "wb") as f:
        pickle.dump(data, f)
    print(data)

if __name__ == "__main__":
    main()
