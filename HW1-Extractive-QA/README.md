# requirements
This project is constructed `under` conda environment

You can set the environment by
```
conda create -n ntu_adl_hw1 python=3.10 -y
conda activate ntu_adl_hw1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn==1.5.1 nltk==3.9.1 tqdm numpy pandas
pip install transformers==4.50.0 datasets==2.21.0 accelerate==0.34.2
pip install evaluate matplotlib gdown
conda install -c conda-forge datasets==2.21.0 pyarrow evaluate
pip install kaggle
```

If you want to reproduce the training / evaluation loss
```
pip install wandb
```

Or you cay run the following to set the environment

```
conda env create -f environment.yml -n ntu_adl_hw1
```

# data download
After setting the kaggle configuration, please run:
```
mkdir ntu_hw1
cd ntu_hw1
mkdir data
cd data
kaggle competitions download -c ntu-adl-2025-hw-1
unzip ntu-adl-2025-hw-1.zip
```

# data_preprocess
please make sure your setting is like:

```
~/where/the/scripts/are$ tree
.
└── data
    ├── context.json
    ├── test.json
    ├── train.json
    └── valid.json
```
* for multiple-choice
```
python step1_mc_preprocess.py
```

* for question-anwer
```
python step1_mc_preprocess.py
```


# model download & unpacking

bash ./download.sh

or you can directly run

```
./download.sh
```

for demo

# Testing

bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv

or you can directly run if you've done `ntu-adl-2025-hw-1` download

```
./run.sh data/context.json data/test.json submission.csv
```

for demo

