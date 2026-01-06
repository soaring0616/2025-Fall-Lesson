# NTU-114-ADL-HW2

This is the execution manual of NTU 114 ADL HW2

# requirements
This project is constructed `under` conda environment

```
conda create -n ntu_hw2 python=3.11 -y
conda activate ntu_hw2
```
- This code is set for the requirements from the `NTU-ADL` class.
- The given `inference` environment is following by the below scripture:

```
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
conda install pyarrow -c conda-forge
pip install "transformers>=4.51.0" bitsandbytes==0.44.1 peft==0.13.0 gdown datasets==3.0.1
```

- This model is train by `Unsloth` on Colab and transformed by `model_trans.py`. 
- If you want to run `train`, please follow the following scripture to set up the environment:

```
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
pip install -q --upgrade transformers peft trl accelerate bitsandbytes scipy wandb
```


## Dataset download
```
!gdown --id 1yFaK4fJRfCeyWBQCehqMFd2S3UzDHywr
!unzip hw2.zip
```


## Train
- I follow the instruction from [this url](https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing).
- You can see the history by `NTU_ADL_HW2_Qwen3_4B_train.ipynb`.

## Inference
Please follow the below instruction to check the submission

### Download
```
bash download.sh
```

### Test w/  output predictions on testing file (.json)
```
bash run.sh \
    /path/to/`Qwen/Qwen3-4B` \
    /path/to/adapter_checkpoint/under/your/folder \
    /path/to/input \
    /path/to/output
```

### example
```
bash run.sh \
    "Qwen/Qwen3-4B"\
    /home/d14948004/adapter_checkpoint \
    /home/d14948004/data/public_test.json \
    /home/d14948004/d14948004_output.json
```

I do it on Colab, which you can see in `NTU_ADL_HW2_Qwen3_4B_inference.ipynb`
