#!/usr/bin/env python3
"""
Food Recognition Calorie Counter - Main Entry Point

A simplified demo version that can work with or without camera/CV2.
Run this file to start the application.
"""

import sys
import time
import random
from typing import Dict, List, Tuple

# Try to import required packages, fall back gracefully
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV (cv2) not available. Installing...")

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_small
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch/torchvision not available. Installing...")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Pillow not available. Installing...")

import numpy as np

from food_database import FoodDatabase
from calorie_estimator import CalorieEstimator


class FoodRecognitionDemo:
    """
    Demo version of Food Recognition App that works with or without camera.
    """
    
    def __init__(self):
        """Initialize the demo app."""
        self.food_db = FoodDatabase()
        self.calorie_estimator = CalorieEstimator()
        self.demo_foods = [
            "apple", "banana", "pizza_slice", "hamburger", "salad",
            "french_fries", "donut", "sandwich", "chicken_breast", "rice"
        ]
        
    def demo_mode(self):
        """Run in demo mode without camera."""
        print("=" * 60)
        print("🍎 FOOD RECOGNITION CALORIE COUNTER - DEMO MODE 🍎")
        print("=" * 60)
        print("\nThis demo simulates food detection and calorie estimation.")
        print("In the full version, this would use your camera feed.")
        print("\nPress Ctrl+C to exit\n")
        
        try:
            while True:
                # Simulate random food detection
                food_name = random.choice(self.demo_foods)
                confidence = random.uniform(0.3, 0.95)
                
                # Get calorie estimation
                calories = self.calorie_estimator.estimate_calories(food_name, confidence)
                
                # Get detailed nutrition info
                nutrition_info = self.calorie_estimator.get_nutrition_info(food_name, confidence)
                
                # Display results
                print(f"🔍 Detected: {food_name.replace('_', ' ').title()}")
                print(f"📊 Confidence: {confidence:.1%}")
                print(f"🔥 Estimated Calories: {calories}")
                print(f"📏 Typical Serving: {nutrition_info['typical_serving_g']}g")
                print(f"📈 Calories per 100g: {nutrition_info['calories_per_100g']}")
                print("-" * 40)
                
                time.sleep(2)  # Wait 2 seconds between detections
                
        except KeyboardInterrupt:
            print("\n👋 Demo ended. Thank you!")
            
    def install_dependencies(self):
        """Install missing dependencies."""
        missing_packages = []
        
        if not CV2_AVAILABLE:
            missing_packages.append("opencv-python")
        if not TORCH_AVAILABLE:
            missing_packages.append("torch torchvision")
        if not PIL_AVAILABLE:
            missing_packages.append("pillow")
            
        if missing_packages:
            print("\n📦 Installing missing dependencies...")
            print("Please run the following command:")
            print(f"pip install {' '.join(missing_packages)}")
            print("\nAfter installation, run this script again for full camera functionality.")
            return False
        return True
        
    def camera_mode(self):
        """Run with camera if available."""
        if not all([CV2_AVAILABLE, TORCH_AVAILABLE, PIL_AVAILABLE]):
            print("❌ Missing dependencies for camera mode.")
            self.install_dependencies()
            return False
            
        print("📹 Starting camera mode...")
        
        # Import the full app (only if dependencies are available)
        try:
            from food_recognition_app import FoodRecognitionApp
            app = FoodRecognitionApp(camera_index=0, confidence_threshold=0.3)
            app.run()
            return True
        except Exception as e:
            print(f"❌ Could not start camera mode: {e}")
            print("🎮 Falling back to demo mode...")
            return False
            
    def show_food_database(self):
        """Display all foods in the database."""
        print("\n" + "=" * 50)
        print("🗂️  FOOD DATABASE")
        print("=" * 50)
        
        foods = self.food_db.get_all_foods()
        for i, food in enumerate(foods, 1):
            info = self.food_db.get_food_info(food)
            if info:
                print(f"{i:2d}. {food.replace('_', ' ').title():<20} "
                      f"({info['calories_per_100g']} cal/100g, "
                      f"~{info['typical_serving']}g serving)")
            else:
                print(f"{i:2d}. {food.replace('_', ' ').title():<20} (Unknown nutrition info)")
                  
        print(f"\nTotal: {len(foods)} food items in database")
        
    def run(self):
        """Main entry point."""
        print("🚀 Starting Food Recognition Calorie Counter...")
        
        # Check if we can run camera mode
        if self.camera_mode():
            return
            
        # Show food database
        self.show_food_database()
        
        # Run demo mode
        print("\n" + "=" * 50)
        input("Press Enter to start demo mode...")
        self.demo_mode()


def main():
    """Main function."""
    demo = FoodRecognitionDemo()
    demo.run()


if __name__ == "__main__":
    main()
