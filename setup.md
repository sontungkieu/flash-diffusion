#1. install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh

#2. create env:
uv sync

#3. download dataset:
uv run makedataset.py

#4. login huggingface:
uv run huggingface-cli login

#5. login wandb:
uv run wandb login

#6. run:
uv run examples/train_flash_sd.py
