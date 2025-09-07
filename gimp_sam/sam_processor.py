#!/usr/bin/env python3

"""
SAM Integration Script for GIMP Plugin
This script handles the actual SAM processing called by the GIMP plugin
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image
import torch

def process_with_fastsam(input_path, output_path, model_type="FastSAM-s"):
    """
    Process image with FastSAM and return segmentation results
    """
    try:
        # Add the parent directory to path to import from object_detection
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(parent_dir, "object_detection"))
        
        # Import FastSAM
        from fastsam import FastSAM, FastSAMPrompt
        
        # Load the model
        model_path = os.path.join(parent_dir, "object_detection", f"{model_type}.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(parent_dir, "food_recognition", f"{model_type}.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_type}.pt not found")
        
        # Initialize FastSAM
        model = FastSAM(model_path)
        
        # Load and process image
        IMAGE_PATH = input_path
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Run inference
        everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # Prepare a prompt processor
        prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
        
        # Everything prompt (segment everything)
        ann = prompt_process.everything_prompt()
        
        # Save the result
        prompt_process.plot(annotations=ann, output_path=output_path)
        
        return True, "Segmentation completed successfully"
        
    except Exception as e:
        return False, f"Error in FastSAM processing: {str(e)}"

def create_masks_from_segmentation(input_path, output_dir, model_type="FastSAM-s"):
    """
    Create individual mask files for each detected object
    """
    try:
        # Add the parent directory to path to import from object_detection
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.join(parent_dir, "object_detection"))
        
        from fastsam import FastSAM, FastSAMPrompt
        
        # Load the model
        model_path = os.path.join(parent_dir, "object_detection", f"{model_type}.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(parent_dir, "food_recognition", f"{model_type}.pt")
        
        model = FastSAM(model_path)
        
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Run inference
        everything_results = model(input_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process results
        if everything_results and len(everything_results) > 0:
            result = everything_results[0]
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                # Save individual masks
                for i, mask in enumerate(masks):
                    mask_path = os.path.join(output_dir, f"mask_{i:03d}.png")
                    
                    # Convert mask to image
                    mask_img = (mask * 255).astype(np.uint8)
                    Image.fromarray(mask_img).save(mask_path)
                
                return True, f"Created {len(masks)} masks in {output_dir}"
            else:
                return False, "No masks found in segmentation results"
        else:
            return False, "No segmentation results"
            
    except Exception as e:
        return False, f"Error creating masks: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="SAM Integration for GIMP")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--output-dir", help="Output directory for masks")
    parser.add_argument("--model", default="FastSAM-s", choices=["FastSAM-s", "FastSAM-x"], help="Model type")
    parser.add_argument("--mode", default="segment", choices=["segment", "masks"], help="Processing mode")
    
    args = parser.parse_args()
    
    if args.mode == "segment":
        if not args.output:
            print("Error: --output is required for segment mode")
            sys.exit(1)
        
        success, message = process_with_fastsam(args.input, args.output, args.model)
        print(message)
        sys.exit(0 if success else 1)
    
    elif args.mode == "masks":
        if not args.output_dir:
            print("Error: --output-dir is required for masks mode")
            sys.exit(1)
        
        success, message = create_masks_from_segmentation(args.input, args.output_dir, args.model)
        print(message)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
