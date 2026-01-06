import json
import random
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Split the train dataset for Retriever training")
parser.add_argument(
        "--train_file",
        type=str,
        default="data/train.txt",
        help="training dataset need to be split",
    )
parser.add_argument(
        "--output_train_file",
        type=str,
        default="data/train_split.txt",
        help="output training dataset",
    )
parser.add_argument(
        "--output_val_file",
        type=str,
        default="data/train_val.txt",
        help="output validation dataset",
    )

parser.add_argument(
        "--ratio",
        type=str,
        default=0.8,
        help="ratio for spliting",
    )

args = parser.parse_args()

random.seed(42)  # reproducibility

# Step 1: Load and group by qid
qid_to_rows = defaultdict(list)
answerable_qids = []
unanswerable_qids = []

with open(args.train_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line.strip())
        qid = item["qid"]
        qid_to_rows[qid].append(item)

# Classify qids
for qid, rows in qid_to_rows.items():
    # Check if any row has answerable text (they should all agree per qid)
    if rows[0]["answer"]["text"] == "CANNOTANSWER":
        unanswerable_qids.append(qid)
    else:
        answerable_qids.append(qid)

print(f"Answerable queries: {len(answerable_qids)}")
print(f"Unanswerable queries: {len(unanswerable_qids)}")

# Step 2: Stratified split
def split_list(lst, ratio=args.ratio):
    n = len(lst)
    split_idx = int(ratio * n)
    random.shuffle(lst)
    return set(lst[:split_idx]), set(lst[split_idx:])

train_ans, val_ans = split_list(answerable_qids, args.ratio)
train_unans, val_unans = split_list(unanswerable_qids, args.ratio)

train_qids = train_ans | train_unans
val_qids = val_ans | val_unans

print(f"\nTrain: {len(train_qids)} queries")
print(f"Val: {len(val_qids)} queries")
print(f"Train answerable ratio: {len(train_ans)/len(train_qids):.2%}")
print(f"Val answerable ratio: {len(val_ans)/len(val_qids):.2%}")

# Step 3: Write splits
with open(args.output_train_file, "w", encoding="utf-8") as f_train, \
     open(args.output_val_file, "w", encoding="utf-8") as f_val:
    for qid, rows in qid_to_rows.items():
        f = f_train if qid in train_qids else f_val
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
