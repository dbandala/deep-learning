"""
Simplified Food Recognition Demo

This version works without requiring model training or complex setup.
It demonstrates the food recognition and calorie estimation concepts.
"""

import cv2
import time
import random
import numpy as np
from typing import Tuple
from food_database import FoodDatabase
from calorie_estimator import CalorieEstimator

class SimpleFoodRecognitionDemo:
    """
    Simplified demo that simulates food recognition using basic computer vision.
    This approach doesn't require ML model training and works immediately.
    """
    
    def __init__(self, camera_index: int = 0):
        """Initialize the demo."""
        self.camera_index = camera_index
        self.food_db = FoodDatabase()
        self.calorie_estimator = CalorieEstimator()
        self.cap = None
        
        # Demo foods that we'll "detect"
        self.demo_foods = [
            "apple", "banana", "orange", "pizza_slice", "hamburger", 
            "sandwich", "salad", "french_fries", "donut", "cookie"
        ]
        
        # Enhanced color-based detection with better food recognition
        self.color_ranges = {
            # Fruits
            "apple": {
                "red": {"lower": np.array([0, 50, 50]), "upper": np.array([10, 255, 255])},
                "green": {"lower": np.array([40, 50, 50]), "upper": np.array([80, 255, 255])}
            },
            "banana": {"lower": np.array([15, 100, 100]), "upper": np.array([25, 255, 255])},
            "orange": {"lower": np.array([8, 100, 100]), "upper": np.array([18, 255, 255])},
            
            # Vegetables
            "salad": {"lower": np.array([40, 50, 50]), "upper": np.array([80, 255, 200])},
            
            # Processed foods (brown/golden colors)
            "bread_slice": {"lower": np.array([10, 50, 100]), "upper": np.array([25, 180, 255])},
            "french_fries": {"lower": np.array([15, 80, 150]), "upper": np.array([25, 255, 255])},
            "donut": {"lower": np.array([8, 100, 120]), "upper": np.array([20, 200, 255])},
            
            # Pizza (reddish due to tomato sauce)
            "pizza_slice": {"lower": np.array([0, 80, 100]), "upper": np.array([15, 255, 255])},
        }
        
        # Shape characteristics for better detection
        self.shape_characteristics = {
            "apple": {"min_area": 2000, "circularity_range": (0.6, 1.0)},
            "banana": {"min_area": 3000, "aspect_ratio_range": (2.5, 5.0)},
            "orange": {"min_area": 2500, "circularity_range": (0.7, 1.0)},
            "pizza_slice": {"min_area": 5000, "aspect_ratio_range": (0.8, 2.0)},
            "french_fries": {"min_area": 1000, "aspect_ratio_range": (3.0, 8.0)},
            "donut": {"min_area": 2000, "circularity_range": (0.7, 1.0)},
            "bread_slice": {"min_area": 3000, "aspect_ratio_range": (0.8, 2.5)},
            "salad": {"min_area": 4000, "circularity_range": (0.3, 0.8)},
        }
        
    def setup_camera(self):
        """Setup camera for demo."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        except:
            return False
    
    def detect_food_advanced(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Advanced food detection using color analysis, shape detection, and contour analysis.
        This uses real computer vision techniques for food recognition.
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        detected_foods = []
        
        for food_name, color_info in self.color_ranges.items():
            masks = []
            
            # Handle multiple color variants (e.g., red and green apples)
            if isinstance(color_info, dict) and "lower" in color_info:
                # Single color range
                mask = cv2.inRange(blurred, color_info["lower"], color_info["upper"])
                masks.append(mask)
            else:
                # Multiple color ranges
                for color_variant, range_info in color_info.items():
                    if isinstance(range_info, dict) and "lower" in range_info:
                        mask = cv2.inRange(blurred, range_info["lower"], range_info["upper"])
                        masks.append(mask)
            
            # Combine all masks for this food type
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Get shape characteristics for this food type
                shape_char = self.shape_characteristics.get(food_name, {"min_area": 1000})
                min_area = shape_char["min_area"]
                
                if area < min_area:
                    continue
                
                # Calculate shape features
                confidence = self._analyze_shape_features(contour, food_name, area)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    detected_foods.append((food_name, confidence, area, contour))
        
        if detected_foods:
            # Sort by confidence and return the best detection
            detected_foods.sort(key=lambda x: x[1], reverse=True)
            best_detection = detected_foods[0]
            return best_detection[0], best_detection[1]
            
        return "No food detected", 0.0
    
    def _analyze_shape_features(self, contour, food_name: str, area: float) -> float:
        """
        Analyze shape features to determine detection confidence.
        """
        # Calculate basic geometric features
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
            
        # Circularity (4π * area / perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Bounding rectangle for aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Get expected characteristics for this food
        shape_char = self.shape_characteristics.get(food_name, {})
        
        confidence = 0.5  # Base confidence
        
        # Check circularity if specified
        if "circularity_range" in shape_char:
            circ_min, circ_max = shape_char["circularity_range"]
            if circ_min <= circularity <= circ_max:
                confidence += 0.3
            else:
                confidence -= 0.2
                
        # Check aspect ratio if specified
        if "aspect_ratio_range" in shape_char:
            ar_min, ar_max = shape_char["aspect_ratio_range"]
            if ar_min <= aspect_ratio <= ar_max:
                confidence += 0.3
            else:
                confidence -= 0.2
        
        # Bonus for larger, well-defined objects
        if area > 5000:
            confidence += 0.1
            
        # Penalty for very irregular shapes (unless it's salad)
        if circularity < 0.3 and food_name not in ["salad", "french_fries"]:
            confidence -= 0.2
            
        return max(0.0, min(1.0, confidence))
    
    def draw_real_time_overlay(self, frame: np.ndarray, food_name: str, 
                         confidence: float, calories: int) -> np.ndarray:
        """Draw real-time detection information overlay on frame."""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Real-time detection indicator
        cv2.putText(frame, "REAL-TIME FOOD DETECTION", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Detection results
        cv2.putText(frame, f"Food: {food_name.replace('_', ' ').title()}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, f"Estimated Calories: {calories}", (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 's' for stats", 
                   (width - 250, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def show_food_stats(self):
        """Show food database statistics."""
        print("\n" + "="*50)
        print("FOOD DATABASE STATISTICS")
        print("="*50)
        
        total_foods = len(self.food_db.get_all_foods())
        print(f"Total food items: {total_foods}")
        
        # Show some examples
        print("\nSample foods and calories:")
        sample_foods = ["apple", "banana", "pizza_slice", "hamburger", "salad"]
        for food in sample_foods:
            info = self.food_db.get_food_info(food)
            if info:
                print(f"  {food.replace('_', ' ').title()}: {info['calories_per_100g']} cal/100g")
        
        print("\nCalorie estimation factors:")
        print("  High confidence (80%+): Full serving")
        print("  Medium confidence (60-80%): 80% of serving") 
        print("  Low confidence (40-60%): 60% of serving")
        print("  Very low confidence (<40%): 40% of serving")
        print("="*50)
    
    def run_camera_demo(self):
        """Run the real-time camera detection demo."""
        if not self.setup_camera():
            print("❌ Could not access camera.")
            print("Please make sure:")
            print("  - Camera is connected and working")
            print("  - No other application is using the camera")
            print("  - Camera permissions are granted")
            return False
            
        print("📹 Real-time food detection started!")
        print("Point your camera at food items for detection")
        print("Supported foods: apple, banana, orange, pizza, salad, bread, fries, donut")
        print("Press 'q' to quit, 's' to show food database stats")
        
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame_count += 1
                
                # Real-time food detection using computer vision
                food_name, confidence = self.detect_food_advanced(frame)
                
                # Count successful detections
                if confidence > 0:
                    detection_count += 1
                
                # Calculate calories
                calories = 0
                if confidence > 0:
                    calories = self.calorie_estimator.estimate_calories(food_name, confidence)
                
                # Draw overlay with detection results
                display_frame = self.draw_real_time_overlay(frame, food_name, confidence, calories)
                
                # Show frame
                cv2.imshow('Real-Time Food Detection', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.show_food_stats()
                    print(f"\nSession stats: {detection_count} detections in {frame_count} frames")
                    
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        finally:
            print(f"\nSession summary: {detection_count} food detections in {frame_count} frames")
            self.cleanup()
            
        return True
    
    def run(self):
        """Main entry point for real-time food detection."""
        print("🍎 REAL-TIME FOOD RECOGNITION 🍎")
        print("="*50)
        print("This application performs real-time food detection using computer vision.")
        print("It uses color analysis, shape detection, and contour analysis.")
        print()
        print("Supported foods:")
        print("🍎 Fruits: Apple, Banana, Orange")
        print("🍕 Fast Food: Pizza Slice") 
        print("🥗 Healthy: Salad")
        print("🍞 Bread Products: Bread Slice, Donut")
        print("🍟 Snacks: French Fries")
        print()
        
        if not self.run_camera_demo():
            print("\n❌ Camera detection failed!")
            print("Make sure you have:")
            print("  - A working camera connected")
            print("  - opencv-python installed: pip install opencv-python")
            print("  - Camera permissions granted")
            return
            
        print("\n✅ Real-time detection completed successfully!")
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function."""
    demo = SimpleFoodRecognitionDemo()
    demo.run()


if __name__ == "__main__":
    main()
