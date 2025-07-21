#!/usr/bin/env python3
"""
FastSAM Object Segmentation Script
A lightweight script for segmenting all objects in an image using FastSAM
Compatible with current ultralytics versions
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import FastSAM
import random

def setup_fastsam():
    """Initialize FastSAM model"""
    # Load pre-trained FastSAM model
    model = FastSAM('FastSAM-s.pt')  # or 'FastSAM-s.pt' for smaller/faster version
    return model

def segment_everything(model, image_path, conf_threshold=0.4, iou_threshold=0.9):
    """
    Segment all objects in an image using the standard ultralytics API
    
    Args:
        model: FastSAM model
        image_path: Path to input image
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    
    Returns:
        results: Segmentation results with masks
    """
    # Run inference
    results = model(
        image_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        retina_masks=True,
        imgsz=1024,
        conf=conf_threshold,
        iou=iou_threshold,
    )
    
    return results

def create_colored_mask(masks, colors=None):
    """
    Create a colored visualization of multiple masks
    
    Args:
        masks: Tensor or numpy array of masks [N, H, W]
        colors: Optional list of colors for each mask
    
    Returns:
        colored_mask: RGB image with colored masks
    """
    if masks is None or len(masks) == 0:
        return None
    
    # Convert to numpy if tensor
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    h, w = masks.shape[-2:]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate random colors if not provided
    if colors is None:
        colors = []
        for _ in range(len(masks)):
            colors.append([random.randint(0, 255) for _ in range(3)])
    
    # Apply each mask with its color
    for i, mask in enumerate(masks):
        if i < len(colors):
            color = colors[i]
            colored_mask[mask > 0.5] = color
    
    return colored_mask

def visualize_results(image_path, results, save_path=None):
    """Visualize segmentation results"""
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # type: ignore
    
    # Create subplot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation results
    axes[1].imshow(image)
    
    if results and len(results) > 0:
        result = results[0]  # Get first result
        
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data  # Get mask data
            colored_mask = create_colored_mask(masks)
            
            if colored_mask is not None:
                axes[1].imshow(colored_mask, alpha=0.6)
            
            axes[1].set_title(f'Segmentation Results ({len(masks)} objects found)')
        else:
            axes[1].set_title('No masks found')
    else:
        axes[1].set_title('No results')
    
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def save_individual_masks(results, image_path, output_dir="masks"):
    """Save individual masks as separate files"""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if results and len(results) > 0:
        result = results[0]
        
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):
                # Convert to 0-255 range
                mask_img = (mask * 255).astype(np.uint8)
                
                # Save mask
                mask_path = os.path.join(output_dir, f"mask_{i:03d}.png")
                cv2.imwrite(mask_path, mask_img)
            
            print(f"Saved {len(masks)} individual masks to {output_dir}/")

def get_largest_masks(results, top_k=5):
    """Get the k largest masks by area"""
    if not results or len(results) == 0:
        return None
    
    result = results[0]
    if not hasattr(result, 'masks') or result.masks is None:
        return None
    
    masks = result.masks.data.cpu().numpy()
    
    # Calculate area for each mask
    areas = [np.sum(mask) for mask in masks]
    
    # Get indices of largest masks
    largest_indices = np.argsort(areas)[-top_k:][::-1]
    
    return masks[largest_indices]

def main():
    """Main execution function"""
    # Setup
    print("Loading FastSAM model...")
    model = setup_fastsam()
    
    # Image path - CHANGE THIS TO YOUR IMAGE
    image_path = "1.jpg"  # Replace with your image path
    
    # Check if image exists
    import os
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        print("Please update the image_path variable with a valid image file.")
        return
    
    print(f"Running segmentation on {image_path}...")
    
    # Run segmentation
    results = segment_everything(model, image_path)
    
    # Print results info
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'masks') and result.masks is not None:
            num_objects = len(result.masks.data)
            print(f"Found {num_objects} objects")
            
            # Print confidence scores if available
            if hasattr(result, 'boxes') and result.boxes is not None:
                if hasattr(result.boxes, 'conf'):
                    confidences = result.boxes.conf.cpu().numpy()
                    print(f"Confidence scores: {confidences}")
        else:
            print("No masks detected")
    else:
        print("No results obtained")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(image_path, results, "segmentation_results.jpg")
    
    # Save individual masks (optional)
    save_individual_masks(results, image_path)
    
    print("Segmentation complete!")

if __name__ == "__main__":
    # Install requirements first:
    # pip install ultralytics opencv-python matplotlib torch torchvision
    main()