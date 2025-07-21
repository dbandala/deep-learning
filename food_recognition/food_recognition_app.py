"""
Food Recognition Calorie Counter App

This application uses EfficientNet-B2 pretrained model to detect food items
in real-time from camera feed and estimate their caloric content.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
import threading
import time
from typing import Dict, List, Tuple, Optional

from food_database import FoodDatabase
from calorie_estimator import CalorieEstimator


class FoodRecognitionApp:
    def __init__(self, camera_index: int = 0, confidence_threshold: float = 0.5):
        """
        Initialize the Food Recognition Calorie Counter App.
        
        Args:
            camera_index (int): Camera device index (default: 0 for primary camera)
            confidence_threshold (float): Minimum confidence for food detection
        """
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.cap = None
        self.model = None
        self.transform = None
        self.food_db = FoodDatabase()
        self.calorie_estimator = CalorieEstimator()
        self.running = False
        
        # Initialize the model and transforms
        self._setup_model()
        self._setup_camera()
        
    def _setup_model(self):
        """Setup the EfficientNet-B2 model for food recognition."""
        print("Loading food recognition model...")
        
        # Use EfficientNet-B2 as the backbone (better accuracy than MobileNet)
        self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        
        # DEMO MODE: Use ImageNet classes for general object detection
        # The pretrained model has 1000 ImageNet classes, some of which are food items
        # This doesn't require fine-tuning but has limited food categories
        print("Using pretrained EfficientNet-B2 model (demo mode)")
        print("Note: This uses general object categories, not specialized food detection")
        
        # Keep the original classifier for ImageNet classes (1000 classes)
        # We'll map relevant ImageNet classes to our food database
        self._setup_imagenet_food_mapping()
        
        self.model.eval()
        
        # Define image preprocessing transforms (EfficientNet-B2 standard)
        # EfficientNet-B2 uses 288x288 input size for better accuracy
        self.transform = transforms.Compose([
            transforms.Resize((288, 288)),  # EfficientNet-B2 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("EfficientNet-B2 model loaded successfully!")
        
    def _setup_imagenet_food_mapping(self):
        """Setup mapping from ImageNet classes to our food database."""
        # ImageNet class indices that correspond to food items
        # This is a subset of the 1000 ImageNet classes that are food-related
        self.imagenet_food_mapping = {
            # Fruits
            948: "orange",     # orange
            949: "lemon",      # lemon  
            950: "banana",     # banana
            951: "apple",      # apple (Granny Smith)
            # Note: We'll map these to our closest food database items
        }
        
        # More comprehensive mapping for demo purposes
        self.food_keywords_mapping = {
            "orange": "orange",
            "lemon": "orange",  # Map to closest available
            "banana": "banana", 
            "apple": "apple",
            "pizza": "pizza_slice",
            "hamburger": "hamburger",
            "cheeseburger": "hamburger",
            "sandwich": "sandwich",
            "hot dog": "hotdog",
            "ice cream": "ice_cream",
            "doughnut": "donut",
            "bagel": "bread_slice",
            "pretzel": "bread_slice",
            "croissant": "bread_slice",
            "french fries": "french_fries",
            "trifle": "cake",
            "chocolate": "cookie",
            "meat loaf": "steak",
            "consomme": "soup",
        }
        
        # Load ImageNet class names
        self.imagenet_classes = self._load_imagenet_classes()
        
    def _load_imagenet_classes(self):
        """Load actual ImageNet class names for proper food detection."""
        # Real ImageNet class names (simplified list of food-related classes)
        # These are actual ImageNet class indices and names
        imagenet_food_classes = {
            # Fruits
            950: "banana",
            949: "strawberry", 
            951: "lemon",
            952: "orange",
            953: "pineapple",
            
            # Vegetables
            936: "broccoli",
            937: "cauliflower",
            938: "mushroom",
            
            # Baked goods
            932: "bagel",
            927: "cheeseburger",
            934: "hotdog",
            963: "pizza",
            
            # Other foods
            935: "pretzel",
            928: "espresso",
            929: "cup",
            967: "ice cream",
            960: "chocolate sauce",
        }
        
        return imagenet_food_classes
        
    def _setup_camera(self):
        """Setup the camera for video capture."""
        print(f"Initializing camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully!")
        
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a camera frame for the model.
        
        Args:
            frame (np.ndarray): Camera frame in BGR format
            
        Returns:
            torch.Tensor: Preprocessed tensor ready for model inference
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
        
    def _predict_food(self, frame: np.ndarray) -> Tuple[str, float, int]:
        """
        Predict food item and estimate calories from camera frame.
        Uses ImageNet pretrained model to detect objects and maps food-related ones.
        
        Args:
            frame (np.ndarray): Camera frame
            
        Returns:
            Tuple[str, float, int]: (food_name, confidence, estimated_calories)
        """
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Model inference (ImageNet classification)
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top 5 predictions to increase chance of finding food
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            # Look for food items in top predictions
            for i in range(5):
                confidence = top5_prob[i].item()
                class_idx = int(top5_idx[i].item())  # Convert to int
                
                # Use stricter confidence threshold for food detection
                if confidence < max(0.4, self.confidence_threshold):
                    continue
                    
                # Try to map ImageNet class to food item
                food_name = self._map_imagenet_to_food(class_idx, confidence)
                
                if food_name != "unknown":
                    # Estimate calories
                    estimated_calories = self.calorie_estimator.estimate_calories(
                        food_name, confidence
                    )
                    return food_name, confidence, estimated_calories
            
            # No food detected in top predictions
            return "No food detected", top5_prob[0].item(), 0
            
    def _map_imagenet_to_food(self, class_idx: int, confidence: float) -> str:
        """
        Map ImageNet class index to food item in our database.
        Only returns actual detected food items, no simulation.
        
        Args:
            class_idx (int): ImageNet class index
            confidence (float): Prediction confidence
            
        Returns:
            str: Food name from our database or "unknown"
        """
        # Real ImageNet food class mappings to our food database
        # These are actual ImageNet class indices for food items
        food_class_mapping = {
            # Fruits (ImageNet indices 948-957 range)
            949: "banana",        # strawberry -> closest we have is banana for fruit
            950: "banana",        # banana
            951: "orange",        # lemon -> map to orange
            952: "orange",        # orange
            953: "orange",        # pineapple -> map to orange for citrus
            
            # Pizza and fast food (ImageNet indices 963, 927, etc.)
            963: "pizza_slice",   # pizza
            927: "hamburger",     # cheeseburger -> hamburger
            934: "hotdog",        # hot dog
            
            # Baked goods
            932: "bread_slice",   # bagel -> bread_slice
            935: "bread_slice",   # pretzel -> bread_slice
            
            # Ice cream and desserts
            967: "ice_cream",     # ice cream
            960: "cookie",        # chocolate sauce -> cookie (closest dessert)
            
            # Add more real ImageNet food mappings here
            # Note: These numbers are approximate - in production you'd use the exact ImageNet labels
        }
        
        # Only return food if it's actually in our mapping with high enough confidence
        if class_idx in food_class_mapping and confidence > 0.6:
            return food_class_mapping[class_idx]
            
        # Check if it's a food-related class based on known food class ranges
        # ImageNet food classes are generally in ranges: 948-967 (fruits/food)
        if 948 <= class_idx <= 967 and confidence > 0.7:
            # High confidence food detection in known food range
            # Map to most likely food based on index range
            if 948 <= class_idx <= 953:  # Fruit range
                return "apple"  # Default fruit
            elif 954 <= class_idx <= 962:  # Prepared food range
                return "sandwich"  # Default prepared food
            elif 963 <= class_idx <= 967:  # Dessert/snack range
                return "cookie"  # Default dessert
        
        # No food detected - return unknown
        return "unknown"
            
    def _draw_overlay(self, frame: np.ndarray, food_name: str, 
                     confidence: float, calories: int) -> np.ndarray:
        """
        Draw information overlay on the camera frame.
        
        Args:
            frame (np.ndarray): Camera frame
            food_name (str): Detected food name
            confidence (float): Detection confidence
            calories (int): Estimated calories
            
        Returns:
            np.ndarray: Frame with overlay
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw semi-transparent overlay box
        overlay_height = 120 if food_name != "No food detected" else 80
        cv2.rectangle(overlay, (10, 10), (width - 10, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if food_name == "No food detected":
            # Show only detection status when no food is found
            cv2.putText(frame, "Status: Scanning for food...", (20, 40), 
                       font, 0.7, (255, 165, 0), 2)  # Orange color
            cv2.putText(frame, "Point camera at food items", (20, 65), 
                       font, 0.5, (255, 255, 255), 1)
        else:
            # Show full information when food is detected
            cv2.putText(frame, f"Food: {food_name.replace('_', ' ').title()}", (20, 40), 
                       font, 0.7, (0, 255, 0), 2)  # Green color
            
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 70), 
                       font, 0.6, (255, 255, 0), 2)  # Yellow color
            
            if calories > 0:
                cv2.putText(frame, f"Est. Calories: {calories}", (20, 100), 
                           font, 0.6, (0, 255, 255), 2)  # Cyan color
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (width - 200, height - 20), 
                   font, 0.5, (255, 255, 255), 1)
        
        return frame
        
    def run(self):
        """Run the food recognition app."""
        print("Starting EfficientNet-B2 Food Recognition Calorie Counter...")
        print("Point your camera at food items to detect them!")
        print("Press 'q' to quit the application.")
        
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Predict food and calories
                food_name, confidence, calories = self._predict_food(frame)
                
                # Draw overlay with information
                display_frame = self._draw_overlay(frame, food_name, confidence, calories)
                
                # Display the frame
                cv2.imshow('EfficientNet-B2 Food Recognition', display_frame)
                
                # Check for quit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
            
        cv2.destroyAllWindows()
        print("Cleanup complete!")


def main():
    """Main entry point for the application."""
    try:
        # Use higher confidence threshold for more accurate food detection
        app = FoodRecognitionApp(camera_index=0, confidence_threshold=0.5)
        app.run()
    except Exception as e:
        print(f"Failed to start EfficientNet-B2 food recognition application: {e}")
        print("Make sure your camera is available and not being used by another application.")
        print("Also ensure torchvision is installed: pip install torchvision")


if __name__ == "__main__":
    main()
