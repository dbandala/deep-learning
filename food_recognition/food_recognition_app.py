"""
Food Recognition Calorie Counter App

This application uses a lightweight PyTorch model to detect food items
in real-time from camera feed and estimate their caloric content.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
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
        """Setup the MobileNetV3 model for food recognition."""
        print("Loading food recognition model...")
        
        # Use MobileNetV3 as the backbone (lightweight model)
        self.model = mobilenet_v3_small(pretrained=True)
        
        # Modify the classifier for food recognition
        # Note: In a real application, you'd fine-tune this on a food dataset
        num_classes = len(self.food_db.get_all_foods())
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.classifier[0].in_features, 1024),
            torch.nn.Hardswish(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, num_classes)
        )
        
        self.model.eval()
        
        # Define image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print("Model loaded successfully!")
        
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
        
        Args:
            frame (np.ndarray): Camera frame
            
        Returns:
            Tuple[str, float, int]: (food_name, confidence, estimated_calories)
        """
        with torch.no_grad():
            # Preprocess frame
            input_tensor = self._preprocess_frame(frame)
            
            # Model inference
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 0)
            confidence = confidence.item()
            
            if confidence < self.confidence_threshold:
                return "No food detected", confidence, 0
                
            # Get food name from database
            food_name = self.food_db.get_food_by_index(predicted_idx.item())
            
            # Estimate calories (simplified estimation based on typical serving)
            estimated_calories = self.calorie_estimator.estimate_calories(
                food_name, confidence
            )
            
            return food_name, confidence, estimated_calories
            
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
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Food name
        cv2.putText(frame, f"Food: {food_name}", (20, 40), 
                   font, 0.7, (0, 255, 0), 2)
        
        # Confidence
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), 
                   font, 0.6, (255, 255, 0), 2)
        
        # Calories
        if calories > 0:
            cv2.putText(frame, f"Est. Calories: {calories}", (20, 100), 
                       font, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (width - 200, height - 20), 
                   font, 0.5, (255, 255, 255), 1)
        
        return frame
        
    def run(self):
        """Run the food recognition app."""
        print("Starting Food Recognition Calorie Counter...")
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
                cv2.imshow('Food Recognition Calorie Counter', display_frame)
                
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
        app = FoodRecognitionApp(camera_index=0, confidence_threshold=0.3)
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Make sure your camera is available and not being used by another application.")


if __name__ == "__main__":
    main()
