from datasets import load_dataset
import webdataset as wds
import json
from PIL import Image
import os

# Tải dataset (500 samples Pokémon)
ds = load_dataset("diffusers/pokemon-gpt4-captions", split="train[:500]")
print(f"Loaded {len(ds)} samples")

# Tạo thư mục lưu shards (nếu chưa có)
output_dir = "pokemon_shards"
os.makedirs(output_dir, exist_ok=True)

# Xóa shards cũ nếu cần (tùy chọn)
# import shutil
# if os.path.exists(output_dir):
#     shutil.rmtree(output_dir)
#     os.makedirs(output_dir)

# Convert sang shard .tar với key đúng
writer = wds.ShardWriter(os.path.join(output_dir, "shards-%05d.tar"), maxsize=1e9)

for i, sample in enumerate(ds):
    img = sample['image']  # PIL.Image
    caption = sample['text']
    
    key = f"{i:08d}"  # "00000000"
    sample_dict = {
        "__key__": key,  # Metadata
        "jpg": img,  # "00000000.jpg"
        "json": json.dumps({"caption": caption}).encode('utf-8')  # "00000000.json"
    }
    
    writer.write(sample_dict)
    if i % 100 == 0:
        print(f"Written {i+1}/{len(ds)} samples")

writer.close()
print(f"Shard created in: {output_dir}/")
