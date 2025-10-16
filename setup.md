install uv:
curl -LsSf https://astral.sh/uv/install.sh | sh
create env:
uv sync
download dataset:
uv run makedataset.py
login huggingface:
uv run huggingface-cli login
login wandb:
uv run wandb login
run:
uv run examples/train_flash_sd.py
