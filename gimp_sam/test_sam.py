#!/usr/bin/env python3

"""
Test script for SAM processor
This script helps test the SAM functionality independently of GIMP
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

def test_sam_processor():
    """Test the SAM processor with a sample image"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    processor_path = script_dir / "sam_processor.py"
    
    if not processor_path.exists():
        print("❌ sam_processor.py not found!")
        return False
    
    # Check if we have any test images in the project
    project_root = script_dir.parent
    test_images = []
    
    # Look for test images
    for img_dir in ["object_detection", "optical_character_recognition"]:
        img_path = project_root / img_dir
        if img_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                test_images.extend(img_path.glob(ext))
    
    if not test_images:
        print("❌ No test images found in project directories")
        return False
    
    # Use the first available test image
    test_image = str(test_images[0])
    print(f"🖼️  Using test image: {test_image}")
    
    # Create temporary output files
    temp_dir = tempfile.mkdtemp()
    output_image = os.path.join(temp_dir, "sam_test_output.jpg")
    output_masks = os.path.join(temp_dir, "masks")
    
    print(f"📁 Temporary output directory: {temp_dir}")
    
    # Test segmentation mode
    print("\n🔄 Testing segmentation mode...")
    import subprocess
    
    cmd_segment = [
        sys.executable,
        str(processor_path),
        "--input", test_image,
        "--output", output_image,
        "--mode", "segment",
        "--model", "FastSAM-s"
    ]
    
    try:
        result = subprocess.run(cmd_segment, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Segmentation mode: SUCCESS")
            if os.path.exists(output_image):
                print(f"   Output saved to: {output_image}")
            else:
                print("⚠️  Output file not created")
        else:
            print("❌ Segmentation mode: FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Segmentation mode: TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"❌ Segmentation mode: ERROR - {e}")
        return False
    
    # Test mask generation mode
    print("\n🔄 Testing mask generation mode...")
    
    cmd_masks = [
        sys.executable,
        str(processor_path),
        "--input", test_image,
        "--output-dir", output_masks,
        "--mode", "masks",
        "--model", "FastSAM-s"
    ]
    
    try:
        result = subprocess.run(cmd_masks, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Mask generation mode: SUCCESS")
            if os.path.exists(output_masks):
                mask_files = [f for f in os.listdir(output_masks) if f.endswith('.png')]
                print(f"   Generated {len(mask_files)} mask files")
            else:
                print("⚠️  Mask directory not created")
        else:
            print("❌ Mask generation mode: FAILED")
            print(f"   Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Mask generation mode: TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"❌ Mask generation mode: ERROR - {e}")
        return False
    
    print(f"\n🧹 Cleaning up temporary files in {temp_dir}")
    import shutil
    shutil.rmtree(temp_dir)
    
    print("\n🎉 All tests passed! The SAM processor is working correctly.")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics")
    ]
    
    missing = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT FOUND")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Install with: pip install torch torchvision ultralytics Pillow numpy")
        return False
    
    print("✅ All dependencies found!")
    return True

def check_models():
    """Check if SAM model files are available"""
    print("\n🔍 Checking for SAM model files...")
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    model_dirs = [
        project_root / "object_detection",
        project_root / "food_recognition"
    ]
    
    models_found = []
    
    for model_dir in model_dirs:
        if model_dir.exists():
            for model_file in ["FastSAM-s.pt", "FastSAM-x.pt"]:
                model_path = model_dir / model_file
                if model_path.exists():
                    models_found.append(str(model_path))
                    print(f"   ✅ {model_file} found in {model_dir.name}")
    
    if not models_found:
        print("   ❌ No SAM model files found!")
        print("   Please download FastSAM-s.pt or FastSAM-x.pt to object_detection/ or food_recognition/")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test SAM GIMP plugin components")
    parser.add_argument("--skip-test", action="store_true", help="Only check dependencies and models")
    
    args = parser.parse_args()
    
    print("🧪 SAM GIMP Plugin Test Suite")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check models
    if not check_models():
        sys.exit(1)
    
    if args.skip_test:
        print("\n✅ Pre-flight checks passed! Ready for testing.")
        return
    
    # Run processor tests
    if test_sam_processor():
        print("\n🎉 SAM processor is ready for use in GIMP!")
    else:
        print("\n❌ SAM processor tests failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
