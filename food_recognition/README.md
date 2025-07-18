# 🍎 Food Recognition Calorie Counter

A real-time food recognition and calorie estimation application using PyTorch and computer vision.

## ⚠️ Real-Time Detection System

**This implementation provides real-time food detection using computer vision.**

### Detection Approach

The current system uses advanced computer vision techniques:

1. **Color Analysis**: Multi-range HSV color detection for different food types
2. **Shape Analysis**: Circularity and aspect ratio analysis for food identification  
3. **Contour Detection**: Morphological operations and contour analysis
4. **Confidence Scoring**: Multi-factor confidence calculation based on shape and size

### Real-Time Features

- **Live Camera Feed**: Processes camera input in real-time
- **Advanced CV Algorithms**: Uses OpenCV for robust detection
- **Multiple Food Types**: Detects 8+ food categories
- **Shape Characteristics**: Analyzes geometric properties for accuracy
- **No Training Required**: Works immediately with computer vision

### For Machine Learning Approach

To create an ML-based food recognition system, you would need to:

1. **Fine-tune on food data**: Use a dataset like Food-101
2. **Transfer learning**: Keep pretrained features, train only classifier
3. **Use food-specific models**: Download models already trained on food

See `MODEL_APPROACHES.md` for detailed ML implementation strategies.

## Features

- **Real-time Food Detection**: Uses computer vision to detect food items in real-time
- **Advanced CV Algorithms**: Color analysis, shape detection, and contour analysis
- **Calorie Estimation**: Estimates calories based on detected food and confidence levels
- **No Training Required**: Uses computer vision techniques, works immediately
- **Food Database**: Contains 25+ common food items with nutritional information
- **Multiple Detection Methods**: Color-based and shape-based detection algorithms

## Installation

1. Make sure you're in the food_recognition directory:
   ```bash
   cd food_recognition
   ```

2. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python torchvision numpy
   ```

## Usage

### Quick Start

Run the main application:
```bash
python main.py
```

### Camera Mode (Full Application)

If all dependencies are installed, the app will automatically start in camera mode:
- Point your camera at food items
- The app will detect food and estimate calories in real-time
- Press 'q' to quit

### Demo Mode

If camera/dependencies are not available, the app will run in demo mode:
- Simulates food detection with random foods from the database
- Shows how calorie estimation works
- Press Ctrl+C to exit

### Running Individual Components

You can also run the full app directly:
```bash
python food_recognition_app.py
```

## Food Database

The app includes a database of 25+ common food items including:

- **Fruits**: Apple, Banana, Orange
- **Fast Food**: Pizza, Hamburger, French Fries
- **Healthy Options**: Salad, Chicken Breast, Fish
- **Snacks**: Donut, Cookie, Ice Cream
- **Meals**: Rice, Pasta, Soup, Sandwich

Each food item includes:
- Calories per 100g
- Typical serving size
- Estimated calories per serving

## How It Works

1. **Camera Input**: Captures frames from your camera
2. **Preprocessing**: Resizes and normalizes images for the model
3. **Food Detection**: Uses MobileNetV3 to classify food items
4. **Confidence Analysis**: Adjusts calorie estimates based on detection confidence
5. **Calorie Estimation**: Calculates calories using food database and serving sizes
6. **Real-time Display**: Shows results with overlay on camera feed

## Model Architecture

- **Base Model**: MobileNetV3-Small (lightweight and efficient)
- **Input Size**: 224x224 RGB images
- **Output**: 25 food classes
- **Inference Time**: ~10-20ms on modern CPUs

## Calorie Estimation Logic

The app estimates calories using:
- Base calories from food database
- Confidence-based adjustments (lower confidence = smaller portion)
- Typical serving sizes for each food type

**Confidence Factors**:
- 80%+ confidence: Full serving (100%)
- 60-80%: Most serving (80%)
- 40-60%: Partial serving (60%)
- 30-40%: Small portion (40%)
- <30%: Very small portion (20%)

## Customization

### Adding New Foods

You can add new foods to the database:

```python
from food_database import FoodDatabase

db = FoodDatabase()
db.add_food("new_food", calories_per_100g=200, typical_serving=100)
```

### Adjusting Model

The model can be fine-tuned on your own food dataset by modifying the classifier in `food_recognition_app.py`.

## Troubleshooting

### Camera Not Working
- Make sure no other application is using the camera
- Try changing the camera index in the code (0, 1, 2, etc.)
- Check camera permissions on your system

### Missing Dependencies
- Run `pip install -r requirements.txt`
- Make sure you have Python 3.8+
- On some systems, you might need `pip3` instead of `pip`

### Performance Issues
- Close other applications using the camera
- Reduce camera resolution if needed
- The model is optimized for real-time inference

## Technical Details

### File Structure
```
food_recognition/
├── main.py                    # Main entry point
├── food_recognition_app.py    # Full camera application
├── food_database.py          # Food database management
├── calorie_estimator.py      # Calorie estimation logic
├── requirements.txt          # Additional dependencies
└── README.md                 # This file
```

### Dependencies
- **PyTorch**: Deep learning framework
- **torchvision**: Pre-trained models and transforms
- **OpenCV**: Camera input and image processing
- **Pillow**: Image manipulation
- **NumPy**: Numerical computations

## Future Improvements

- [ ] Fine-tune model on food-specific dataset
- [ ] Add portion size estimation using object detection
- [ ] Implement food nutrition tracking over time
- [ ] Add more food categories and items
- [ ] Support for multiple food items in one frame
- [ ] Integration with nutrition APIs for more accurate data

## License

This project is for educational and personal use. The pre-trained models are subject to their respective licenses.

## Contributing

Feel free to contribute by:
- Adding new food items to the database
- Improving calorie estimation algorithms
- Enhancing the user interface
- Adding new features

---

**Note**: This is a demonstration application. For production use, consider fine-tuning the model on a dedicated food dataset and implementing more sophisticated portion size estimation.
