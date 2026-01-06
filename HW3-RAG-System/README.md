# Requirements
This project is constructed `under` conda environment

```
conda create -n ntu_hw3 python=3.12 -y
conda activate ntu_hw3
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install transformers==4.56.1
pip install datasets==4.0.0
pip install tqdm==4.67.1
pip install faiss-gpu==1.12.0
pip install sentence-transformers==5.1.0
pip install python-dotenv==1.1.1
pip install accelerate==1.10.1
pip install gdown
```

This project is log by `wandb`.
Please make sure you have log in wandb


# Data download
Please download the relevant data here:
https://drive.google.com/drive/folders/1v5hSQYPyQuUnzaE1Lp3F1vejNazW48TH

# Inference

## Download the models
please run
```
bash download.sh
```



## Build vector database
Building sqlite DB to store passages, Faiss vector DB to store embeddings
```
python save_embeddings.py --retriever_model_path /path/to/retriever/ --build_db
```

## Run


```
python inference_batch.py --retriever_model_path /path/to/retriever/ --reranker_model_path /path/to/reranker/ --test_data_path /path/to/test/data
```



# Train

## Retriever
Please format the dataset by

```
python split_train_val.py /path/to/original/dataset /path/to/train/split /path/to/val/split ratio=custom_ratio
```

for example if the train:val ratio should be 8:2:

```
python split_train_val.py data/train.txt data/train_split.txt data/train_val.txt ratio=0.8
```

Execute train by

```
python retriever_1.py /path/to/pretrained/model /path/to/corpus/ /path/to/qrels/ /path/to/train/ /path/to/val/ /path/to/output/dir
```

for example if the

```
python retriever_1.py "intfloat/multilingual-e5-small" data/corpus.txt data/qrels.txt data/train_split.txt data/train_val.txt ./models/retriever/
```

## Reranker
Please format the dataset by
```
python reranker_data.py /path/to/corpus/ /path/to/retriever/model/ /path/to/train/split /path/to/val/split /path/to/train/reranker/ /path/to/val/reranker top_k=k
```

for example:

```
python reranker_data.py data/corpus.txt ./models/retriever/ data/train_split.txt data/train_val.txt data/reranker_train.jsonl data/reranker_val.jsonl top_k=10
```

Execute train by

```
python reranker_5.py /path/to/train/reranker/ /path/to/val/reranker /path/to/output/dir
``` 

for example:

```
python reranker_5.py data/reranker_train.jsonl data/reranker_val.jsonl ./models/reranker/
```



# Misc.
- `q3_analyze.py` is for the analysis of Q3 in the report. This file should be revised in the main-loop (this version need the output files (results/results.json))
- The table below is for the comparison of different `utils.py` and prompts, which shows in the report.

| Files | Prompt | 
| -------- | -------- | 
| utils_1.py     | Prompt 1     | 
| utils_2.py     | Prompt 2     | 
| utils_3.py     | Prompt 3     | 
| utils_4.py     | Prompt 4     | 
| utils_5.py     | Prompt 5     | 
| utils_6.py     | Prompt 6     | 
| utils.py     | Prompt 7     | 
