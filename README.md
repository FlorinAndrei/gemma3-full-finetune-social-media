Full fine-tuning of the Gemma3 LLM using a dataset with social media comments.

The following process works well on an NVIDIA DGX Spark with 20 ARM cores, 128 GB unified RAM, running Ubuntu 24.04 LTS, with CUDA-13.0 installed on bare metal.

# Install dependencies

Run all commands in the folder containing this repository.

Install Python packages and build tools:

```
apt-get update
apt-get install python3-pip python3-venv python3-dev build-essential
```

Create and activate the Python environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch with CUDA-13.0 modules - this will install several `nvidia-*` modules as well:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If the following command returns `True` then you have the right combination of PyTorch and CUDA versions, and the flash attention module will compile successfully:

```
python -c "import torch; print(torch.cuda._is_compiled())"
```

If the command returns `False`, either fix the version combo, or do not install flash attention (some operations with the model will run more slowly and will use more RAM). The wrong combination of PyTorch and CUDA versions will trigger this error while compiling flash attention:

```
CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```

Even if you set CUDA_HOME with the right value, the error will persist if there is a version mismatch.

Install most modules:

```
pip install -r requirements.txt
```

Compile and install the flash attention module (this will take a long time - the wheel file will have over 230 MB in size):

```
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

If memory usage during compilation is less than 50%, increase MAX_JOBS to speed up the process. If the system runs out of memory, decrease MAX_JOBS.

Download (pre-cache) a few base models:

```
hf download --max-workers 1 google/gemma-3-27b-it
hf download --max-workers 1 google/gemma-3-12b-it
hf download --max-workers 1 google/gemma-3-4b-it
```

# Dataset

The dataset I used is all comments I made on a large social media site since the day I started using it many years ago. It has about 30k observations. Each observation is a prompt/answer pair: the prompt is the comment I was responding to, and the answer is my own comment.

This dataset is not public. But any prompt/answer dataset should work, if it's big enough. Put it in a file called `conversations.csv` with two columns: `parent_text` (the prompts) and `comment_body` (the answers). It's okay if each comment spans multiple rows, as long as the proper file format is used.

See the file `conversations-template.txt` for an illustration of the desired format.

# Training

You do not have to always run training in Jupyter. The script `run-training.sh` can be used to run training as a detached process on a headless machine. The script collects stdout and stderr from training in two log files. You can simply ssh into the system and run:

```
bash run-training.sh train.ipynb
```

Then log out and let the training process continue to completion.

`run-tensorboard.sh` can be used to start tensorboard as a background process, listening on all interfaces.
