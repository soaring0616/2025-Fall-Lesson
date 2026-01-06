import json
import os
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import logging
import wandb
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a bi-encoder model on a Retriever task [rag]")
    parser.add_argument(
        "--retriever_train_dataset",
        type=str,
        default="data/train_split.txt",
        help="The path to the retriever training dataset",
    )
    parser.add_argument(
        "--retriever_eval_dataset",
        type=str,
        default="data/train_val.txt",
        help="The path to the retriever validation dataset",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/corpus.txt",
        help="The path to corpus",
    )
    parser.add_argument(
        "--qrel",
        type=str,
        default="data/qrel.txt",
        help="The path to qrel",
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        default="./models/retriever/",
        help="Where to store the final model",
    )

    args = parser.parse_args()

    return args




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === some config ===
args = parse_args()

model_name = "intfloat/multilingual-e5-small"
output_dir = args.output_model_dir
os.makedirs(output_dir, exist_ok=True)

CORPUS_PATH = args.corpus
QRELS_PATH = args.qrels
TRAIN_PATH = args.retriever_train_dataset
VAL_PATH = args.retriever_eval_dataset

num_epochs = 5
train_batch_size = 128
use_fp16 = True
learning_rate = 1e-5
eval_steps = 40

# === Data Loading ===
logger.info("Loading corpus...")
corpus = {}
with open(CORPUS_PATH) as f:
    for line in f:
        doc = json.loads(line.strip())
        corpus[doc["id"]] = f"passage: {doc['text']}"

logger.info("Loading qrels...")
qrels = {}
with open(QRELS_PATH) as f:
    qrels_data = json.load(f)
    for qid, pos_dict in qrels_data.items():
        qrels[qid] = next(iter(pos_dict.keys()))

logger.info("Building training examples...")
train_examples = []
with open(TRAIN_PATH) as f:
    for line in f:
        item = json.loads(line.strip())
        qid = item["qid"]
        if qid in qrels and qrels[qid] in corpus:
            query = f"query: {item['rewrite']}"
            pos = corpus[qrels[qid]]
            train_examples.append(InputExample(texts=[query, pos]))

# === model ===
word_embedding_model = models.Transformer(model_name, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_cls_token=True)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train_loss = losses.MultipleNegativesRankingLoss(model)
device = model.device

# === evaluator ===
ir_evaluator = None
if os.path.exists(VAL_PATH):
    val_queries, val_qrels, val_corpus_ids = {}, {}, set()
    with open(VAL_PATH) as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item["qid"]
            if qid not in val_queries:
                val_queries[qid] = f"query: {item['rewrite']}"
                if qid in qrels:
                    pos_id = qrels[qid]
                    val_qrels[qid] = {pos_id: 1}
                    val_corpus_ids.add(pos_id)
    val_corpus = {cid: corpus[cid] for cid in val_corpus_ids if cid in corpus}
    if val_queries and val_corpus and val_qrels:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=val_queries,
            corpus=val_corpus,
            relevant_docs=val_qrels,
            score_functions={"cos_sim": cos_sim},
            name="val_ir",
            batch_size=128,
        )

# === WandB ===
wandb.init(project="retriever-rag", config={
    "model": model_name,
    "batch_size": train_batch_size,
    "lr": learning_rate,
    "epochs": num_epochs,
})
wandb.watch(model)

# === 訓練 loop ===
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size, collate_fn=model.smart_batching_collate)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
global_step = 0
best_recall10 = 0.0

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        global_step += 1

        # to GPU
        features, labels = batch
        features = [  # features --> list of dict
            {k: v.to(model.device) for k, v in f.items()} for f in features
        ]

        with torch.amp.autocast('cuda', enabled=use_fp16):
            loss = train_loss(features, labels=None)


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        wandb.log({"train/loss": loss.item(), "step": global_step})

        # Evaluation
        if ir_evaluator and global_step % eval_steps == 0:
            model.eval()
            #scores = ir_evaluator(model, output_path=None, epoch=epoch, steps=global_step)
            scores = ir_evaluator.compute_metrices(model)
            model.train()
            
            wandb.log({
                "val/recall@1": scores["cos_sim"]["recall@k"][1],
                "val/recall@5": scores["cos_sim"]["recall@k"][5],
                "val/recall@10": scores["cos_sim"]["recall@k"][10],
                "val/mrr@10": scores["cos_sim"]["mrr@k"][10],
                "step": global_step,
            })

            if scores["cos_sim"]["recall@k"][10] > best_recall10:
                best_recall10 = scores["cos_sim"]["recall@k"][10]
                model.save(os.path.join(output_dir, "best_model"))

            
    if ir_evaluator:
        model.eval()
        scores = ir_evaluator.compute_metrices(model)
        model.train()

        
        wandb.log({
                "val/recall@1": scores["cos_sim"]["recall@k"][1],
                "val/recall@5": scores["cos_sim"]["recall@k"][5],
                "val/recall@10": scores["cos_sim"]["recall@k"][10],
                "val/mrr@10": scores["cos_sim"]["mrr@k"][10],
                "step": global_step,
        })

        if scores["cos_sim"]["recall@k"][10] > best_recall10:
            best_recall10 = scores["cos_sim"]["recall@k"][10]
            model.save(os.path.join(output_dir, "best_model"))

model.save(os.path.join(output_dir, "final_model"))
wandb.finish()
logger.info("Training finished.")
