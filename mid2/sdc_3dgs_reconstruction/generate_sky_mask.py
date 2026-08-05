import argparse
import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _preprocess(image: Image.Image, size=(640, 640)) -> dict:
    img = image.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return {"pixel_values": tensor}

class SkySegmenter:
    def __init__(self, device="cuda"):
        self.device = device
        model_id = "nvidia/segformer-b5-finetuned-ade-640-640"
        print(f"[Init] Loading {model_id}...")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def process_image(self, image_path, output_path):
        image = Image.open(image_path).convert("RGB")
        # Preprocess
        inputs = {k: v.to(self.device) for k, v in _preprocess(image).items()}
        # Inference
        outputs = self.model(**inputs)
        logits = outputs.logits  # [1, 150, H/4, W/4]
        
        # Resize to original image size
        upscaled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        
        # ADE20K label index 2 is "Sky"
        prediction = torch.argmax(upscaled_logits, dim=1).squeeze(0).cpu().numpy()
        mask = (prediction == 2).astype(np.uint8) * 255
        
        # Save binary mask
        cv2.imwrite(output_path, mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_num", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Track root directory. Defaults to mid2dataset/Track{N}")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = f"mid2dataset/Track{args.track_num}"

    input_dir = os.path.join(args.data_dir, "data", "image")
    output_dir = os.path.join(args.data_dir, "data", "sky_masks")

    if not os.path.exists(input_dir):
        print(f"[Error] Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[Info] Found {len(img_files)} images in {input_dir}")

    if len(img_files) == 0:
        print("[Error] No images found. Check your 'input_dir' path.")
        return

    segmenter = SkySegmenter()

    for img_name in tqdm(img_files, desc="Processing Sky Masks"):
        in_path = os.path.join(input_dir, img_name)
        # Save as .png to avoid compression artifacts
        out_name = os.path.splitext(img_name)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        segmenter.process_image(in_path, out_path)
    
    print(f"[Done] Masks saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    main()