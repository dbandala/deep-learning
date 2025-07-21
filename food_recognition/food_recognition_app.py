"""
Food Recognition Calorie Counter App with FastSAM Segmentation

This application uses FastSAM (Fastest Segment Anything Model) to segment food items
in real-time from camera feed and estimate their caloric content.
"""

import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import FastSAM
import threading
import time
import random
from typing import Dict, List, Tuple, Optional

from food_database import FoodDatabase
from calorie_estimator import CalorieEstimator


class FoodRecognitionApp:
    def __init__(self, camera_index: int = 0, confidence_threshold: float = 0.4, show_all_masks: bool = False):
        """
        Initialize the Food Recognition Calorie Counter App with FastSAM.
        
        Args:
            camera_index (int): Camera device index (default: 0 for primary camera)
            confidence_threshold (float): Minimum confidence for food detection
            show_all_masks (bool): Show all segmented objects, not just food items
        """
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.show_all_masks = show_all_masks
        self.cap = None
        self.model = None
        self.food_db = FoodDatabase()
        self.calorie_estimator = CalorieEstimator()
        self.running = False
        
        # Initialize the model and camera
        self._setup_model()
        self._setup_camera()
        
    def _setup_model(self):
        """Setup the FastSAM model for food segmentation."""
        print("Loading FastSAM model...")
        
        # Use FastSAM-s (smallest/fastest model) for real-time segmentation
        try:
            # Look for the model file in the object_detection directory
            model_path = "../object_detection/FastSAM-s.pt"
            self.model = FastSAM(model_path)
            print("Using local FastSAM-s.pt model")
        except:
            # Fallback to downloading the model
            print("Local model not found, downloading FastSAM-s...")
            self.model = FastSAM('FastSAM-s.pt')
        
        print("FastSAM model loaded successfully!")
        print("Model will segment all objects and identify food items")
        
        # Setup food detection parameters
        self.iou_threshold = 0.7
        self.min_mask_area = 1000  # Minimum area for a valid food segment
        
        # Simple food classification based on color and shape patterns
        self._setup_food_heuristics()
        
    def _setup_food_heuristics(self):
        """Setup advanced heuristics to classify segmented objects as food."""
        # Enhanced color ranges with better HSV tuning
        self.food_color_ranges = {
            'fruits': {
                'orange': [(8, 120, 120), (28, 255, 255)],   # Oranges, carrots, peppers
                'red': [(0, 120, 100), (15, 255, 255)],      # Apples, tomatoes, strawberries  
                'yellow': [(20, 100, 120), (40, 255, 255)],  # Bananas, lemons, corn
                'green': [(40, 50, 50), (80, 255, 255)],     # Green apples, lettuce, broccoli
                'purple': [(120, 50, 50), (140, 255, 255)],  # Grapes, eggplant, cabbage
            },
            'cooked_food': {
                'brown': [(8, 50, 30), (25, 200, 180)],      # Bread, meat, chocolate
                'golden': [(15, 80, 120), (35, 255, 255)],   # Fried foods, pastries
                'dark_brown': [(5, 80, 20), (20, 255, 120)], # Coffee, dark chocolate, meat
            },
            'dairy': {
                'white': [(0, 0, 180), (180, 40, 255)],      # Milk, cheese, yogurt
                'cream': [(15, 20, 180), (45, 100, 255)],    # Cream, butter, vanilla
                'yellow_white': [(45, 30, 200), (60, 120, 255)], # Cheese, butter
            },
            'vegetables': {
                'bright_green': [(60, 100, 100), (80, 255, 255)], # Lettuce, spinach
                'dark_green': [(40, 120, 50), (70, 255, 200)],    # Broccoli, kale
                'red_veg': [(0, 100, 100), (10, 255, 255)],       # Tomatoes, red peppers
            }
        }
        
        # Enhanced shape patterns with more food categories
        self.food_shape_patterns = {
            'round_fruits': {'min_circularity': 0.6, 'aspect_ratio_range': (0.7, 1.3)},
            'elongated_food': {'min_circularity': 0.2, 'aspect_ratio_range': (0.2, 0.6)}, # Bananas, carrots
            'rectangular_food': {'min_circularity': 0.3, 'aspect_ratio_range': (1.5, 4.0)}, # Bread, sandwiches
            'irregular_food': {'min_circularity': 0.1, 'aspect_ratio_range': (0.4, 2.5)}, # Pizza, salads
            'small_round': {'min_circularity': 0.7, 'aspect_ratio_range': (0.8, 1.2)},   # Grapes, berries
        }
        
        # Enhanced food mapping with more categories
        self.pattern_to_food = {
            # Fruits
            ('fruits', 'orange', 'round_fruits'): 'orange',
            ('fruits', 'red', 'round_fruits'): 'apple', 
            ('fruits', 'yellow', 'round_fruits'): 'orange',  # Yellow citrus
            ('fruits', 'yellow', 'elongated_food'): 'banana',
            ('fruits', 'green', 'round_fruits'): 'apple',
            ('fruits', 'purple', 'small_round'): 'apple',    # Grapes -> apple (closest)
            
            # Vegetables  
            ('vegetables', 'bright_green', 'irregular_food'): 'salad',
            ('vegetables', 'dark_green', 'round_fruits'): 'apple',  # Broccoli -> apple
            ('vegetables', 'red_veg', 'round_fruits'): 'apple',     # Tomato -> apple
            
            # Cooked foods
            ('cooked_food', 'brown', 'rectangular_food'): 'bread_slice',
            ('cooked_food', 'golden', 'round_fruits'): 'donut',
            ('cooked_food', 'golden', 'irregular_food'): 'pizza_slice',
            ('cooked_food', 'dark_brown', 'irregular_food'): 'steak',
            
            # Dairy
            ('dairy', 'white', 'round_fruits'): 'ice_cream',
            ('dairy', 'cream', 'irregular_food'): 'ice_cream',
            ('dairy', 'yellow_white', 'rectangular_food'): 'cheese',
        }
        
        # Add texture analysis parameters
        self.texture_features = {
            'smooth_threshold': 10,      # Low texture variance for smooth foods
            'rough_threshold': 50,       # High texture variance for rough foods
            'edge_density_threshold': 0.1  # Edge density for complex textures
        }
        
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
        
    def _segment_food_items(self, frame: np.ndarray) -> List[Tuple[str, float, int, np.ndarray]]:
        """
        Segment food items from camera frame using FastSAM.
        
        Args:
            frame (np.ndarray): Camera frame in BGR format
            
        Returns:
            List[Tuple[str, float, int, np.ndarray]]: List of (food_name, confidence, calories, mask)
        """
        try:
            # Run FastSAM segmentation
            results = self.model(
                frame,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                retina_masks=True,
                imgsz=640,  # Smaller size for real-time performance
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
            )
            
            detected_foods = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    # Process each mask
                    for i, mask in enumerate(masks):
                        # Check mask size (filter out tiny segments)
                        mask_area = np.sum(mask > 0.5)
                        if mask_area < self.min_mask_area:
                            continue
                        
                        if self.show_all_masks:
                            # Show all segmented objects with generic label
                            food_name = f"object_{i+1}"
                            confidence = 0.5
                            calories = 0  # No calorie estimate for non-food objects
                            detected_foods.append((food_name, confidence, calories, mask))
                        else:
                            # Classify this segment as food using heuristics
                            food_info = self._classify_food_segment(frame, mask)
                            
                            if food_info['is_food']:
                                food_name = food_info['food_type']
                                confidence = food_info['confidence']
                                
                                # Estimate calories based on segment size and food type
                                calories = self.calorie_estimator.estimate_calories_from_mask(
                                    food_name, mask, frame.shape
                                )
                                
                                detected_foods.append((food_name, confidence, calories, mask))
            
            return detected_foods
            
        except Exception as e:
            print(f"Error in food segmentation: {e}")
            return []
    
    def _classify_food_segment(self, frame: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Classify a segmented region as food using color and shape heuristics.
        
        Args:
            frame (np.ndarray): Original frame
            mask (np.ndarray): Segmentation mask
            
        Returns:
            Dict: Classification results with food type and confidence
        """
        # Extract the segmented region
        mask_bool = mask > 0.5
        
        if not np.any(mask_bool):
            return {'is_food': False, 'food_type': 'unknown', 'confidence': 0.0}
        
        # Get the masked region
        masked_region = frame.copy()
        masked_region[~mask_bool] = 0
        
        # Convert to HSV for better color analysis
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_masked = hsv_frame.copy()
        hsv_masked[~mask_bool] = 0
        
        # Analyze colors in the segmented region
        dominant_color_category = self._analyze_colors(hsv_masked, mask_bool)
        
        # Analyze shape properties
        shape_category = self._analyze_shape(mask_bool)
        
        # Analyze texture properties
        texture_category = self._analyze_texture(masked_region, mask_bool)
        
        # Combine color, shape, and texture to classify food
        food_classification = self._combine_features_for_food_classification(
            dominant_color_category, shape_category, texture_category
        )
        
        return food_classification
    
    def _analyze_colors(self, hsv_frame: np.ndarray, mask: np.ndarray) -> str:
        """Analyze the dominant colors in a masked region."""
        if not np.any(mask):
            return 'unknown'
        
        # Get HSV values from the masked region
        h_values = hsv_frame[mask, 0]
        s_values = hsv_frame[mask, 1] 
        v_values = hsv_frame[mask, 2]
        
        # Calculate mean HSV
        mean_h = np.mean(h_values) if len(h_values) > 0 else 0
        mean_s = np.mean(s_values) if len(s_values) > 0 else 0
        mean_v = np.mean(v_values) if len(v_values) > 0 else 0
        
        # Check against food color ranges
        for category, colors in self.food_color_ranges.items():
            for color_name, (lower, upper) in colors.items():
                if (lower[0] <= mean_h <= upper[0] and
                    lower[1] <= mean_s <= upper[1] and
                    lower[2] <= mean_v <= upper[2]):
                    return f"{category}_{color_name}"
        
        return 'unknown_color'
    
    def _analyze_shape(self, mask: np.ndarray) -> str:
        """Analyze shape properties of a segmented region."""
        # Find contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 'unknown_shape'
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape properties
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 'unknown_shape'
        
        # Circularity (4π * area / perimeter²)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        
        # Bounding rectangle for aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Classify based on shape patterns
        for pattern_name, properties in self.food_shape_patterns.items():
            min_circularity = properties['min_circularity']
            aspect_range = properties['aspect_ratio_range']
            
            if (circularity >= min_circularity and
                aspect_range[0] <= aspect_ratio <= aspect_range[1]):
                return pattern_name
        
        return 'irregular_shape'
    
    def _analyze_texture(self, masked_region: np.ndarray, mask: np.ndarray) -> str:
        """Analyze texture properties of a segmented region."""
        if not np.any(mask):
            return 'unknown_texture'
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture variance (smoothness)
        masked_gray = gray[mask]
        if len(masked_gray) == 0:
            return 'unknown_texture'
        
        texture_variance = float(np.var(masked_gray.astype(np.float32)))
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges[mask] > 0)
        total_pixels = np.sum(mask)
        edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
        
        # Classify texture
        if texture_variance < self.texture_features['smooth_threshold']:
            return 'smooth'  # Ice cream, yogurt, smooth fruits
        elif texture_variance > self.texture_features['rough_threshold']:
            return 'very_rough'  # Bread crust, fried foods
        elif edge_density > self.texture_features['edge_density_threshold']:
            return 'textured'  # Pizza, salads, complex foods
        else:
            return 'medium_rough'  # Most fruits, cooked vegetables
    
    def _combine_features_for_food_classification(self, color_category: str, 
                                                shape_category: str, texture_category: Optional[str] = None) -> Dict:
        """Combine color, shape, and texture features to classify food."""
        # Parse color category
        if '_' in color_category:
            color_family, color_name = color_category.split('_', 1)
        else:
            color_family, color_name = 'unknown', 'unknown'
        
        # Look for matching pattern (basic color + shape)
        pattern_key = (color_family, color_name, shape_category)
        
        if pattern_key in self.pattern_to_food:
            food_type = self.pattern_to_food[pattern_key]
            base_confidence = 0.8  # High confidence for exact pattern match
            
            # Adjust confidence based on texture if available
            if texture_category:
                texture_bonus = self._get_texture_confidence_bonus(food_type, texture_category)
                confidence = min(0.95, base_confidence + texture_bonus)
            else:
                confidence = base_confidence
                
            return {'is_food': True, 'food_type': food_type, 'confidence': confidence}
        
        # Enhanced partial matches with texture consideration
        partial_matches = {
            'fruits': {
                'smooth': {'food_type': 'apple', 'confidence': 0.7},
                'medium_rough': {'food_type': 'orange', 'confidence': 0.6},
                'textured': {'food_type': 'banana', 'confidence': 0.5},
            },
            'cooked_food': {
                'very_rough': {'food_type': 'bread_slice', 'confidence': 0.7},
                'textured': {'food_type': 'pizza_slice', 'confidence': 0.6},
                'smooth': {'food_type': 'cake', 'confidence': 0.5},
            },
            'dairy': {
                'smooth': {'food_type': 'ice_cream', 'confidence': 0.8},
                'medium_rough': {'food_type': 'cheese', 'confidence': 0.6},
            },
            'vegetables': {
                'textured': {'food_type': 'salad', 'confidence': 0.6},
                'medium_rough': {'food_type': 'apple', 'confidence': 0.5},  # Generic veg -> apple
            }
        }
        
        # Try texture-based matching first
        if texture_category and color_family in partial_matches:
            texture_matches = partial_matches[color_family]
            if texture_category in texture_matches:
                match_info = texture_matches[texture_category]
                return {'is_food': True, **match_info}
        
        # Fallback to basic color family matching
        basic_matches = {
            'fruits': {'food_type': 'apple', 'confidence': 0.4},
            'cooked_food': {'food_type': 'bread_slice', 'confidence': 0.4},
            'dairy': {'food_type': 'ice_cream', 'confidence': 0.4},
            'vegetables': {'food_type': 'apple', 'confidence': 0.3},
        }
        
        if color_family in basic_matches:
            match_info = basic_matches[color_family]
            return {'is_food': True, **match_info}
        
        # Default: not classified as food
        return {'is_food': False, 'food_type': 'unknown', 'confidence': 0.0}
    
    def _get_texture_confidence_bonus(self, food_type: str, texture_category: str) -> float:
        """Get confidence bonus based on texture matching expected food texture."""
        texture_expectations = {
            'apple': ['smooth', 'medium_rough'],
            'orange': ['medium_rough', 'textured'],
            'banana': ['smooth', 'medium_rough'],
            'bread_slice': ['very_rough', 'textured'],
            'pizza_slice': ['textured', 'medium_rough'],
            'ice_cream': ['smooth'],
            'cake': ['smooth', 'medium_rough'],
            'donut': ['medium_rough', 'textured'],
            'cheese': ['smooth', 'medium_rough'],
            'salad': ['textured'],
        }
        
        if food_type in texture_expectations:
            expected_textures = texture_expectations[food_type]
            if texture_category in expected_textures:
                return 0.1  # 10% confidence bonus for matching texture
            else:
                return -0.1  # 10% penalty for mismatched texture
        
        return 0.0  # No bonus for unknown food types
            
    def _draw_overlay(self, frame: np.ndarray, detected_foods: List[Tuple[str, float, int, np.ndarray]]) -> np.ndarray:
        """
        Draw information overlay and segmentation masks on the camera frame.
        
        Args:
            frame (np.ndarray): Camera frame
            detected_foods (List): List of (food_name, confidence, calories, mask) tuples
            
        Returns:
            np.ndarray: Frame with overlay and masks
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw segmentation masks with different colors and better visibility
        if detected_foods:
            # Create a separate mask overlay for better control
            mask_overlay = np.zeros_like(frame)
            
            for i, (food_name, confidence, calories, mask) in enumerate(detected_foods):
                # Generate a unique, bright color for each food item
                colors = [
                    [0, 255, 0],      # Bright green
                    [255, 0, 0],      # Bright blue (BGR)
                    [0, 0, 255],      # Bright red
                    [255, 255, 0],    # Cyan
                    [255, 0, 255],    # Magenta  
                    [0, 255, 255],    # Yellow
                    [128, 255, 0],    # Spring green
                    [255, 128, 0],    # Orange
                    [128, 0, 255],    # Purple
                    [0, 128, 255],    # Light blue
                ]
                
                color = colors[i % len(colors)]  # Cycle through colors
                
                # Apply colored mask
                mask_bool = mask > 0.5
                mask_overlay[mask_bool] = color
                
                # Draw contour outline for better visibility
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, color, 2)  # Draw thick contour
                
                # Add food label near the center of the mask
                if np.any(mask_bool):
                    # Find centroid of the mask
                    moments = cv2.moments(mask_uint8)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        
                        # Draw food name with background
                        label = f"{food_name.replace('_', ' ').title()}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        thickness = 2
                        
                        # Get text size for background rectangle
                        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw background rectangle
                        cv2.rectangle(frame, 
                                    (cx - text_width//2 - 5, cy - text_height - 5),
                                    (cx + text_width//2 + 5, cy + 5),
                                    (0, 0, 0), -1)  # Black background
                        
                        # Draw text
                        cv2.putText(frame, label, 
                                  (cx - text_width//2, cy), 
                                  font, font_scale, color, thickness)
            
            # Blend mask overlay with original frame for semi-transparent effect
            alpha = 0.4  # Transparency level (0.0 = transparent, 1.0 = opaque)
            cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw information box
        info_height = max(100, 30 + len(detected_foods) * 25) if detected_foods else 80
        cv2.rectangle(frame, (10, 10), (width - 10, info_height), (0, 0, 0), -1)
        cv2.addWeighted(frame[10:info_height, 10:width-10], 0.7, 
                       np.zeros((info_height-10, width-20, 3), dtype=np.uint8), 0.3, 0,
                       frame[10:info_height, 10:width-10])
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 35
        
        if not detected_foods:
            cv2.putText(frame, "Status: Scanning for food...", (20, y_offset), 
                       font, 0.7, (255, 165, 0), 2)  # Orange color
            cv2.putText(frame, "Point camera at food items", (20, y_offset + 25), 
                       font, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Found {len(detected_foods)} food item(s):", (20, y_offset), 
                       font, 0.6, (0, 255, 0), 2)  # Green color
            
            y_offset += 30
            total_calories = 0
            
            for i, (food_name, confidence, calories, _) in enumerate(detected_foods):
                display_name = food_name.replace('_', ' ').title()
                text = f"{i+1}. {display_name}: {calories} cal ({confidence:.1%})"
                cv2.putText(frame, text, (20, y_offset), 
                           font, 0.5, (255, 255, 255), 1)
                y_offset += 20
                total_calories += calories
            
            # Show total calories
            if len(detected_foods) > 1:
                cv2.putText(frame, f"Total: {total_calories} calories", (20, y_offset), 
                           font, 0.6, (0, 255, 255), 2)  # Cyan color
        
        # Instructions with new controls
        cv2.putText(frame, "Press 'q' to quit, 'm' to toggle mask mode", (width - 350, height - 20), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Show current mode
        mode_text = "ALL OBJECTS" if self.show_all_masks else "FOOD ONLY"
        cv2.putText(frame, f"Mode: {mode_text}", (width - 200, 30), 
                   font, 0.6, (255, 255, 0), 2)
        
        return frame
        
    def run(self):
        """Run the FastSAM food recognition app."""
        print("Starting FastSAM Food Recognition Calorie Counter...")
        print("The app will segment all objects and identify food items!")
        print("Press 'q' to quit the application.")
        
        self.running = True
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Segment food items using FastSAM
                detected_foods = self._segment_food_items(frame)
                
                # Draw overlay with segmentation masks and information
                display_frame = self._draw_overlay(frame, detected_foods)
                
                # Display the frame
                cv2.imshow('FastSAM Food Recognition', display_frame)
                
                # Check for quit key and mode toggle
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    # Toggle mask display mode
                    self.show_all_masks = not self.show_all_masks
                    mode = "ALL OBJECTS" if self.show_all_masks else "FOOD ONLY"
                    print(f"Switched to mode: {mode}")
                    
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
    import sys
    
    # Check for debug mode argument
    show_all_masks = "--show-all" in sys.argv or "--debug" in sys.argv
    
    try:
        # Use moderate confidence threshold for food segmentation
        print(f"Starting FastSAM Food Recognition...")
        if show_all_masks:
            print("DEBUG MODE: Showing all segmented objects")
        else:
            print("FOOD MODE: Showing only detected food items")
            
        app = FoodRecognitionApp(
            camera_index=0, 
            confidence_threshold=0.4,
            show_all_masks=show_all_masks
        )
        app.run()
    except Exception as e:
        print(f"Failed to start FastSAM food recognition application: {e}")
        print("Make sure your camera is available and not being used by another application.")
        print("Also ensure ultralytics is installed: pip install ultralytics")
        print("FastSAM model will be downloaded automatically on first run.")
        print("\nUsage:")
        print("  python food_recognition_app.py              # Show only food items")
        print("  python food_recognition_app.py --show-all   # Show all segmented objects")
        print("  python food_recognition_app.py --debug      # Same as --show-all")


if __name__ == "__main__":
    main()
