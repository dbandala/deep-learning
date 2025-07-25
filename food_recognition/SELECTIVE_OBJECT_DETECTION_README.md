# Selective Object Segmentation App

A real-time object detection and segmentation application using FastSAM (Fastest Segment Anything Model) that allows you to specify which objects to detect using natural language descriptions.

## Features

- **Selective Detection**: Only detects and segments objects you specify using text prompts
- **Real-time Performance**: Optimized frame processing for smooth real-time operation
- **Interactive Configuration**: Change target objects during runtime by pressing 'r'
- **Advanced Preprocessing**: Enhanced frame preprocessing for better detection accuracy
- **Visual Feedback**: High-quality segmentation masks with confidence scores
- **Performance Monitoring**: Real-time performance metrics display

## Installation

1. Make sure you have the required dependencies:
```bash
pip install ultralytics opencv-python torch numpy pillow matplotlib
```

2. The FastSAM model will be downloaded automatically on first run, or you can place `FastSAM-s.pt` in the `../object_detection/` directory.

## Usage

### Basic Usage

Run the app with default objects (person, cup, laptop, cell phone, book):
```bash
python selective_object_segmentation_app.py
```

### Custom Objects

Specify your own target objects:
```bash
python selective_object_segmentation_app.py car,bicycle,person,dog
```

### Interactive Mode

During runtime, you can:
- Press **'r'** to change target objects
- Press **'q'** to quit the application

### Demo Suite

Run the comprehensive demo to see different configurations:
```bash
python selective_demo.py
```

## Configuration Options

The `SelectiveObjectSegmentationApp` class accepts several parameters:

```python
app = SelectiveObjectSegmentationApp(
    target_objects=["person", "car", "dog"],     # Objects to detect
    camera_index=0,                              # Camera device index
    confidence_threshold=0.5,                    # Detection confidence
    process_every_n_frames=10,                   # Frame skip rate for performance
    stabilization_frames=4                       # Frames needed for stable results
)
```

### Parameters Explained

- **target_objects**: List of object names/descriptions to detect (e.g., ["person", "car", "laptop"])
- **camera_index**: Camera device index (0 for primary camera)
- **confidence_threshold**: Minimum confidence score for detections (0.0-1.0)
- **process_every_n_frames**: Process every Nth frame to improve performance
- **stabilization_frames**: Number of consecutive similar results needed before updating display

## Object Types

The app can detect a wide variety of objects. Some examples:

### Common Objects
- person, car, bicycle, motorcycle, airplane, bus, train, truck
- traffic light, fire hydrant, stop sign, parking meter, bench
- cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

### Household Items
- laptop, mouse, remote, keyboard, cell phone, microwave, oven
- toaster, sink, refrigerator, book, clock, vase, scissors
- teddy bear, hair drier, toothbrush, chair, couch, bed, table

### Food Items
- banana, apple, sandwich, orange, broccoli, carrot, hot dog
- pizza, donut, cake, chair, wine glass, cup, fork, knife, spoon

### Sports & Recreation
- frisbee, skis, snowboard, sports ball, kite, baseball bat
- baseball glove, skateboard, surfboard, tennis racket

## Performance Optimization

The app includes several performance optimizations:

1. **Frame Skipping**: Only processes every Nth frame for detection while displaying all frames
2. **Result Stabilization**: Requires multiple consistent detections before updating display
3. **Preprocessing**: Advanced frame enhancement for better detection accuracy
4. **Buffer Management**: Automatic camera buffer clearing when processing is slow

## Technical Details

### Frame Preprocessing
The app applies several computer vision techniques to enhance detection:
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral filtering for noise reduction
- Sharpening filters for edge enhancement
- Gamma correction for lighting adaptation

### Detection Pipeline
1. Capture frame from camera
2. Apply preprocessing enhancements
3. Run FastSAM with text prompts for target objects
4. Filter results by confidence and mask area
5. Apply result stabilization
6. Draw segmentation masks and labels
7. Display with performance information

## Troubleshooting

### Common Issues

1. **Camera not found**: Make sure no other application is using the camera
2. **Poor detection**: Try adjusting the `confidence_threshold` parameter
3. **Slow performance**: Increase `process_every_n_frames` for better speed
4. **Model download fails**: Check internet connection for first-time model download

### Performance Tips

- Use specific object names for better detection (e.g., "laptop" instead of "computer")
- Ensure good lighting conditions for optimal detection
- Keep target object list reasonable (5-10 objects) for best performance
- Adjust confidence threshold based on your needs (lower = more detections, higher = more accurate)

## Examples

### Example 1: Office Environment
```python
target_objects = ["laptop", "monitor", "keyboard", "mouse", "chair", "person"]
```

### Example 2: Kitchen Detection
```python
target_objects = ["microwave", "refrigerator", "sink", "cup", "plate", "spoon"]
```

### Example 3: Outdoor Scene
```python
target_objects = ["car", "person", "bicycle", "tree", "building"]
```

### Example 4: Pet Detection
```python
target_objects = ["cat", "dog"]
```

## API Reference

### Main Class: SelectiveObjectSegmentationApp

#### Methods:
- `__init__(target_objects, camera_index=0, confidence_threshold=0.5, ...)`: Initialize the app
- `run()`: Start the main detection loop
- `update_target_objects(new_targets)`: Change target objects during runtime
- `cleanup()`: Clean up resources

#### Key Internal Methods:
- `_segment_target_objects(frame)`: Perform object detection and segmentation
- `_preprocess_frame(frame)`: Apply frame enhancements
- `_draw_overlay(frame, detected_objects)`: Draw segmentation masks and labels
- `_results_are_similar(current, previous)`: Check result stability

## License

This project uses the FastSAM model from Ultralytics. Please refer to their license terms for commercial usage.
