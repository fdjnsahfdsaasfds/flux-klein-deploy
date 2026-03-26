import os
import tempfile
import requests
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from huggingface_hub import snapshot_download
from diffusers import Flux2KleinPipeline

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


class Predictor(BasePredictor):
    def setup(self) -> None:
        print(f"Downloading/Loading {MODEL_ID} weights...")

        # Enable ultra-fast downloading
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # Securely pulls the token from the Replicate environment variable
        hf_token = os.environ.get("HF_TOKEN")

        snapshot_download(
            repo_id=MODEL_ID,
            ignore_patterns=["*.pt", "*.bin"],
            token=hf_token
        )

        print("Loading pipeline into VRAM...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for the image generation"),
        reference_image: Path = Input(
            description="Optional: Input image for image-to-image editing.",
            default=None
        ),
        image_strength: float = Input(
            description="How much to transform the reference image (0.0 to 1.0). Only used if reference_image is provided.",
            default=0.5
        ),

        # LoRA 1
        lora_1_url: str = Input(
            description="Optional: URL to a .safetensors LoRA file (e.g., from CivitAI).",
            default=None
        ),
        lora_1_scale: float = Input(description="Scale/strength for LoRA 1.", default=1.0),

        # LoRA 2
        lora_2_url: str = Input(
            description="Optional: URL to a 2nd .safetensors LoRA file.",
            default=None
        ),
        lora_2_scale: float = Input(description="Scale/strength for LoRA 2.", default=1.0),

        # LoRA 3
        lora_3_url: str = Input(
            description="Optional: URL to a 3rd .safetensors LoRA file.",
            default=None
        ),
        lora_3_scale: float = Input(description="Scale/strength for LoRA 3.", default=1.0),

        civitai_token: str = Input(
            description="Optional: CivitAI API token for downloading gated LoRAs.",
            default=None
        ),
        num_inference_steps: int = Input(description="Number of denoising steps.", default=4),
        guidance_scale: float = Input(description="Guidance scale.", default=3.5),
        seed: int = Input(description="Random seed. Set to -1 to randomize.", default=-1),
    ) -> Path:

        active_adapters = []
        adapter_weights = []
        downloaded_paths = []
        temp_dir = tempfile.mkdtemp()

        def process_lora(url, scale, index):
            if not url:
                return
            print(f"Downloading LoRA {index} from {url}...")
            lora_path = os.path.join(temp_dir, f"lora_{index}.safetensors")

            download_url = url
            if civitai_token:
                delimiter = "&" if "?" in download_url else "?"
                download_url += f"{delimiter}token={civitai_token}"

            response = requests.get(download_url, stream=True, allow_redirects=True)
            response.raise_for_status()

            with open(lora_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            adapter_name = f"custom_lora_{index}"
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            active_adapters.append(adapter_name)
            adapter_weights.append(scale)
            downloaded_paths.append(lora_path)

        # 1. Process all LoRAs
        process_lora(lora_1_url, lora_1_scale, 1)
        process_lora(lora_2_url, lora_2_scale, 2)
        process_lora(lora_3_url, lora_3_scale, 3)

        if active_adapters:
            print(f"Activating LoRAs: {active_adapters} with weights {adapter_weights}")
            self.pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)
        else:
            self.pipe.unload_lora_weights()

        # 2. Setup seed
        if seed == -1:
            seed = int(torch.randint(0, 1000000, (1,)).item())
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # 3. Build inference kwargs
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        # 4. Handle reference image (image-to-image)
        if reference_image:
            print("Processing reference image for unified editing...")
            img = Image.open(str(reference_image)).convert("RGB")
            kwargs["image"] = img
            kwargs["strength"] = image_strength

        # 5. Generate
        print(f"Generating image with seed {seed}...")
        output_image = self.pipe(**kwargs).images[0]

        # 6. Cleanup LoRAs + temp files to free VRAM
        if active_adapters:
            self.pipe.unload_lora_weights()
            for path in downloaded_paths:
                if os.path.exists(path):
                    os.remove(path)

        # 7. Save and return
        out_path = "/tmp/output.jpg"
        output_image.save(out_path, format="JPEG", quality=95)
        return Path(out_path)
