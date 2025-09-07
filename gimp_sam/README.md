# GIMP SAM Plugin Installation Guide

This plugin integrates Segment Anything Model (SAM) functionality into GIMP for advanced image segmentation.

## Prerequisites

1. **GIMP 2.10 or later** with Python support
2. **Python 3.7+** 
3. **Required Python packages**:
   - torch
   - torchvision  
   - ultralytics
   - Pillow
   - numpy

## Installation Steps

### 1. Install Python Dependencies

First, make sure you have the required Python packages installed in your system Python or GIMP's Python environment:

```bash
pip install torch torchvision ultralytics Pillow numpy
```

### 2. Locate GIMP's Plugin Directory

**On macOS:**
```
~/Library/Application Support/GIMP/2.10/plug-ins/
```

**On Linux:**
```
~/.config/GIMP/2.10/plug-ins/
```

**On Windows:**
```
%APPDATA%\GIMP\2.10\plug-ins\
```

### 3. Install the Plugin

1. Create a new folder called `sam_segmentation` in GIMP's plug-ins directory
2. Copy both `sam_segmentation.py` and `sam_processor.py` to this folder
3. Make the plugin executable (Linux/macOS):
   ```bash
   chmod +x sam_segmentation.py
   ```

### 4. Download SAM Models

Make sure you have the FastSAM model files in your project:
- `FastSAM-s.pt` (small model, faster)
- `FastSAM-x.pt` (large model, more accurate)

These should be in either:
- `/path/to/your/project/object_detection/`
- `/path/to/your/project/food_recognition/`

## Usage

1. **Start GIMP**
2. **Open an image**
3. **Access the plugin** via menu: `Filters → AI → SAM Segmentation...` or `Filters → AI → SAM Auto Mask...`

### Available Functions:

- **SAM Segmentation**: Performs general segmentation and overlays results
- **SAM Auto Mask**: Creates individual mask layers for each detected object

## Troubleshooting

### Common Issues:

1. **"Module not found" errors**:
   - Ensure all Python dependencies are installed
   - Check that GIMP is using the correct Python interpreter

2. **"Model not found" errors**:
   - Verify FastSAM model files are in the correct directories
   - Check file permissions

3. **Plugin doesn't appear in menu**:
   - Restart GIMP after installation
   - Check that the plugin file is executable
   - Look in GIMP's error console for Python errors

### Testing the Installation:

1. Open GIMP
2. Create a new image or open an existing one
3. Go to `Filters → AI`
4. You should see "SAM Segmentation..." and "SAM Auto Mask..." options

## File Structure

```
GIMP/2.10/plug-ins/sam_segmentation/
├── sam_segmentation.py      # Main GIMP plugin
└── sam_processor.py         # SAM processing backend
```

## Configuration

You can modify the plugin behavior by editing `sam_segmentation.py`:

- Change default model type
- Adjust confidence thresholds
- Modify output file naming
- Add new processing modes

## Support

If you encounter issues:

1. Check GIMP's Python console for error messages
2. Verify all dependencies are correctly installed
3. Ensure model files are accessible
4. Test the `sam_processor.py` script independently from command line

## Example Command Line Test

Test the processor independently:

```bash
python sam_processor.py --input /path/to/image.jpg --output /path/to/result.jpg --mode segment
```

This should help verify that the SAM processing works before testing in GIMP.
