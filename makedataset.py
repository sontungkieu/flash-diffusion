from datasets import load_dataset
import webdataset as wds
import json
from PIL import Image
import os

# Tải dataset (500 samples Pokémon)
ds = load_dataset("diffusers/pokemon-gpt4-captions", split="train[:500]")
print(f"Loaded {len(ds)} samples")

# Tạo thư mục

# Xóa shards cũ nếu có
# Convert sang shard .tar với key đúng
writer = wds.ShardWriter("pokemon_shards/shards-%05d.tar", maxsize=1e9)
for i, sample in enumerate(ds):
    img = sample['image']  # PIL.Image
    caption = sample['text']
    
    key = f"{i:08d}"  # "00000000"
    sample_dict = {
        "__key__": key,  # Metadata, không lưu file
        "jpg": img,  # "00000000.jpg"
        "json": json.dumps({"caption": caption}).encode('utf-8')  # "00000000.json"
    }
    
    writer.write(sample_dict)
    if i % 100 == 0:
        print(f"Written {i+1}/500 samples")

writer.close()
print("Shard created: /pokemon_shards/shards-00000.tar")
