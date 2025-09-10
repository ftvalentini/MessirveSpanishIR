

### Install conda:
mkdir -p ~/miniconda3 &&
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh &&
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 &&
rm -rf ~/miniconda3/miniconda.sh &
~/miniconda3/bin/conda init bash &&
conda list

### Requirements
conda create -n spanish-ir python=3.10 -y && conda activate spanish-ir
# only for pyserini:
## Source: https://github.com/castorini/pyserini/blob/master/docs/installation.md#mac
conda install wget -y &&
conda install -c conda-forge openjdk=21 maven -y &&
conda install -c conda-forge lightgbm nmslib -y &&
conda install -c pytorch faiss-cpu -y
# choose cuda version according to nvidia-smi & https://pytorch.org/get-started/locally/
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
# For training with deepspeed (requires nvcc with same version as torch.version.cuda)
conda install --force-reinstall -c nvidia cuda-compiler=12.1 cuda-toolkit=12.1
## to debug installation e.g. ERROR: /usr/bin/ld: cannot find -lcudart, try:
# pip install --no-cache-dir deepspeed==0.15.1
# scripts/set_cuda_env_vars.sh
# ll /home/XYZ/miniconda3/envs/spanish-ir/lib/libcud* # check if libcudart.so exists and points to the right file

# Install requirements:
pip install -r requirements.txt

### Generate HF access token in https://huggingface.co/settings/tokens, and save as env var:
echo "export HF_FULL_TOKEN='...'" >> ~/.bashrc
source ~/.bashrc

### Set API keys as environment variables:
echo "export OPENAI_SPANISH_IR_API_KEY='...'" >> ~/.bashrc
source ~/.bashrc

### Others (run in project dir)
mkdir -p logs
mkdir -p runs/plots
chmod +x scripts/*.sh
