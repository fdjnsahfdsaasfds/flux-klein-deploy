import os
from huggingface_hub import snapshot_download

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
hf_token = os.environ.get("HF_TOKEN")

print(f"Downloading {MODEL_ID} weights to bake into the image...")
snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.pt", "*.bin"],
    token=hf_token
)
print("Download complete.")
