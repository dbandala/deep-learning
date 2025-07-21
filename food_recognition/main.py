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
        
    def get_detection_tips(self):
        """Provide tips for better food detection."""
        print("\n" + "=" * 60)
        print("TIPS FOR BETTER FOOD DETECTION")
        print("=" * 60)
        print("🔍 Detection Tips:")
        print("  • Ensure good lighting")
        print("  • Place food on contrasting background")
        print("  • Keep food item centered in camera view")
        print("  • Avoid shadows and reflections")
        print("  • Make sure food item is clearly visible")
        print()
        print("🍎 Best Results With:")
        print("  • Single food items (not mixed plates)")
        print("  • Well-lit environments")
        print("  • Food items that match our color profiles")
        print("  • Clear, unobstructed views")
        print("=" * 60)
            
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
            app = FoodRecognitionApp(camera_index=0, confidence_threshold=0.6)
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
        
        # Show menu
        print("\n" + "=" * 60)
        print("FOOD RECOGNITION OPTIONS")
        print("=" * 60)
        print("1. Real-Time Camera Detection (computer vision)")
        print("2. PyTorch Camera Demo (requires training for accuracy)")
        print("3. Show Food Database")
        print("4. Detection Tips")
        print()
        
        choice = input("Enter your choice (1-4) or press Enter for option 1: ").strip()
        
        if choice == "2":
            # Try PyTorch camera mode (will fall back if issues)
            if self.camera_mode():
                return
            print("🎮 Falling back to real-time detection...")
        elif choice == "3":
            self.show_food_database()
            return
        elif choice == "4":
            self.get_detection_tips()
            input("\nPress Enter to continue...")
            # Show the menu again
            self.run()
            return
        
        # Default: Real-time camera detection
        try:
            from simple_demo import SimpleFoodRecognitionDemo
            real_time_demo = SimpleFoodRecognitionDemo()
            real_time_demo.run()
            return
        except ImportError as e:
            print(f"Real-time detection not available: {e}")
            print("Please install opencv-python: pip install opencv-python")
        except Exception as e:
            print(f"Error running real-time detection: {e}")
        
        # Fallback: show database info
        print("\nFalling back to food database information:")
        self.show_food_database()


def main():
    """Main function."""
    demo = FoodRecognitionDemo()
    demo.run()


if __name__ == "__main__":
    main()
