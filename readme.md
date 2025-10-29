## Setup environment

> note: Please refer to [FAQ](#FAQ) section for any issues.



```bash

# Create virtual environment
conda create -n adv-bmt "python=3.10" -y
conda activate adv-bmt


# Install Metadrive
git clone https://github.com/metadriverse/metadrive.git
cd ~/metadrive
pip install -e .
cd ~/

# Install ScenarioNet
git clone https://github.com/metadriverse/scenarionet.git
cd ~/scenarionet
pip install -e .
cd ~/


# Clone the code to local and Install basic dependency for this project
git clone https://github.com/Yuxin45/Adv-BMT.git
pip install -e .

# Install Waymo Open Dataset
pip uninstall -y waymo-open-dataset-tf-2-11-0
pip uninstall -y waymo-open-dataset-tf-2-12-0
pip install waymo-open-dataset-tf-2-12-0==1.6.4

# Verify pytorch, expect True.
python -c "import torch;print(torch.cuda.is_available())"


# (Optional) If your torch is not installed properly.
# That is, torch.cuda.is_available() is False, then:
# Install pytorch by yourself to make them compatible with your GPU: https://pytorch.org/
# Note: First checkout which cuda you have at your 
ls /usr/local
# For cuda 11.7:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
# For cuda 11.8:
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# For cuda 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

```


## Run experiment

```bash
bash scripts/020...
```

## Evaluate
Add `--eval` flag after for `train_motion.py`.



## FAQ

Q: `ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant`

A: 
```bash
pip install chardet
```

---

Q: `AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)`

A: 
```bash
pip install -U --force-reinstall charset-normalizer
```


---

Q: `RuntimeError: The detected CUDA version (10.1) mismatches the version that was used to compile
PyTorch (11.7). Please make sure to use the same CUDA versions.`

A: Try:
```bash
export CUDA_HOME=/usr/local/cuda-11.7
python setup.py develop
```

---


Q: When compiling MTR's CUDA code (e.g. `python setup.py`) locally: `RuntimeError: The detected CUDA version (11.7) mismatches the version that was used to compile
PyTorch (12.1). Please make sure to use the same CUDA versions.`

A: Try:
```bash
# Download CUDA first: https://developer.nvidia.com/cuda-12-1-0-download-archive

export CUDA_HOME=/usr/local/cuda-12.1
python setup.py develop
```