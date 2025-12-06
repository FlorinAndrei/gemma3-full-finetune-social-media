python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install --upgrade llmcompressor
pip install torch==2.8.0 --force-reinstall --index-url https://download.pytorch.org/whl/cu129

python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
