import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "sam2_hiera_tiny.pt" #sam2.1_hiera_large.pt
model_cfg = "configs/sam2/sam2_hiera_t.yaml" #sam2.1/sam2.1_hiera_l.yaml
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cpu"))

# read image 1.jpg and convert to RGB
from PIL import Image

image = Image.open("1.jpg").convert("RGB")

with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
    predictor.set_image(image)
    # Example input prompts, replace with actual prompts as needed
    point_coords = torch.tensor([[100, 150], [200, 250]])  # Example points
    point_labels = torch.tensor([1, 0])  # Example labels (1 for foreground, 0 for background)

    masks, _, _ = predictor.predict(point_coords.numpy(), point_labels.numpy(), multimask_output=True)

    # Save the first mask as an example
    mask_image = Image.fromarray((masks[0] * 255).astype('uint8'))
    mask_image.save("output_mask.png")

    # pritn the shape of the masks
    print("Masks shape:", masks.shape)
    # print the dtype of the masks
    print("Masks dtype:", masks.dtype)