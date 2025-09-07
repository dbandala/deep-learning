#!/usr/bin/env python3

"""
SAM Segmentation GIMP Plugin
A simple plugin to perform image segmentation using Segment Anything Model (SAM)
"""

from gimpfu import *
import os
import sys
import tempfile

def sam_segmentation(image, drawable, model_type="FastSAM-s"):
    """
    Main function for SAM segmentation in GIMP
    
    Args:
        image: The GIMP image object
        drawable: The current layer/drawable
        model_type: Type of SAM model to use
    """
    
    try:
        # Start an undo group so the operation can be undone in one step
        pdb.gimp_image_undo_group_start(image)
        
        # Get image dimensions
        width = drawable.width
        height = drawable.height
        
        # Create a temporary file to save the current layer
        temp_dir = tempfile.gettempdir()
        temp_input = os.path.join(temp_dir, "gimp_sam_input.png")
        temp_output = os.path.join(temp_dir, "gimp_sam_output.png")
        
        # Export current layer to temporary file
        pdb.file_png_save(image, drawable, temp_input, temp_input, 0, 9, 1, 1, 1, 1, 1)
        
        # Call SAM processing script
        sam_script_path = os.path.join(os.path.dirname(__file__), "sam_processor.py")
        
        if os.path.exists(sam_script_path):
            # Run SAM segmentation
            import subprocess
            cmd = [
                sys.executable, 
                sam_script_path, 
                "--input", temp_input,
                "--output", temp_output,
                "--model", model_type,
                "--mode", "segment"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(temp_output):
                # Load the segmented result back into GIMP
                new_layer = pdb.gimp_file_load_layer(image, temp_output)
                pdb.gimp_image_insert_layer(image, new_layer, None, 0)
                pdb.gimp_item_set_name(new_layer, "SAM Segmentation")
                
                # Clean up temporary files
                if os.path.exists(temp_input):
                    os.remove(temp_input)
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
                pdb.gimp_displays_flush()
                pdb.gimp_message("SAM segmentation completed successfully!")
            else:
                pdb.gimp_message("Error: SAM segmentation failed. " + result.stderr)
        else:
            pdb.gimp_message("Error: SAM script not found at " + sam_script_path)
            
    except Exception as e:
        pdb.gimp_message("Error in SAM segmentation: " + str(e))
    finally:
        # End the undo group
        pdb.gimp_image_undo_group_end(image)

def sam_auto_mask(image, drawable):
    """
    Automatically create masks for all detected objects
    """
    try:
        pdb.gimp_image_undo_group_start(image)
        
        # Get image dimensions
        width = drawable.width
        height = drawable.height
        
        # Create temporary files
        temp_dir = tempfile.gettempdir()
        temp_input = os.path.join(temp_dir, "gimp_sam_mask_input.png")
        masks_dir = os.path.join(temp_dir, "gimp_sam_masks")
        
        # Export current layer to temporary file
        pdb.file_png_save(image, drawable, temp_input, temp_input, 0, 9, 1, 1, 1, 1, 1)
        
        # Call SAM mask generation
        sam_script_path = os.path.join(os.path.dirname(__file__), "sam_processor.py")
        
        if os.path.exists(sam_script_path):
            import subprocess
            cmd = [
                sys.executable,
                sam_script_path,
                "--input", temp_input,
                "--output-dir", masks_dir,
                "--mode", "masks"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(masks_dir):
                # Load each mask as a separate layer
                mask_files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
                mask_files.sort()
                
                for i, mask_file in enumerate(mask_files[:10]):  # Limit to 10 masks
                    mask_path = os.path.join(masks_dir, mask_file)
                    try:
                        mask_layer = pdb.gimp_file_load_layer(image, mask_path)
                        pdb.gimp_image_insert_layer(image, mask_layer, None, 0)
                        pdb.gimp_item_set_name(mask_layer, f"SAM Mask {i+1}")
                    except:
                        continue
                
                # Clean up
                import shutil
                if os.path.exists(masks_dir):
                    shutil.rmtree(masks_dir)
                    
                pdb.gimp_message(f"Created {len(mask_files)} SAM masks!")
            else:
                pdb.gimp_message("Error in SAM mask generation: " + result.stderr)
        else:
            pdb.gimp_message("Error: SAM processor script not found")
        
        # Clean up input file
        if os.path.exists(temp_input):
            os.remove(temp_input)
            
        pdb.gimp_displays_flush()
        
    except Exception as e:
        pdb.gimp_message("Error in SAM auto-masking: " + str(e))
    finally:
        pdb.gimp_image_undo_group_end(image)

# Register the main segmentation function
register(
    "python_fu_sam_segmentation",
    "Perform SAM segmentation on the current layer",
    "Uses Segment Anything Model to segment objects in the image",
    "Your Name",
    "Your Name",
    "2024",
    "SAM Segmentation...",
    "*",
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None),
        (PF_OPTION, "model_type", "SAM Model Type", 0, ["FastSAM-s", "FastSAM-x", "SAM-b"])
    ],
    [],
    sam_segmentation,
    menu="<Image>/Filters/AI/"
)

# Register the auto-mask function
register(
    "python_fu_sam_auto_mask",
    "Automatically create masks for all objects",
    "Uses SAM to automatically detect and mask all objects in the image",
    "Your Name", 
    "Your Name",
    "2024",
    "SAM Auto Mask...",
    "*",
    [
        (PF_IMAGE, "image", "Input image", None),
        (PF_DRAWABLE, "drawable", "Input drawable", None)
    ],
    [],
    sam_auto_mask,
    menu="<Image>/Filters/AI/"
)

main()
