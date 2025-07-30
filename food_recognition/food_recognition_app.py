"""
FastSAM Object Detection App with Food Recognition

This application uses FastSAM (Fastest Segment Anything Model) to detect and segment 
all objects in real-time from camera feed. Food items are automatically identified 
and their caloric content is estimated.
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
    def __init__(self, camera_index: int = 0, confidence_threshold: float = 0.4, show_all_masks: bool = True, 
                 process_every_n_frames: int = 16, stabilization_frames: int = 6):
        """
        Initialize the FastSAM Object Detection App.
        
        Args:
            camera_index (int): Camera device index (default: 0 for primary camera)
            confidence_threshold (float): Minimum confidence for object detection
            show_all_masks (bool): Always True now - shows all detected objects
            process_every_n_frames (int): Process every Nth frame for detection (default: 5)
            stabilization_frames (int): Number of frames to keep results stable (default: 3)
        """
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.show_all_masks = show_all_masks
        self.process_every_n_frames = process_every_n_frames
        self.stabilization_frames = stabilization_frames
        self.cap = None
        self.model = None
        self.food_db = FoodDatabase()
        self.calorie_estimator = CalorieEstimator()
        self.running = False
        
        # Frame processing tracking
        self.frame_count = 0
        self.last_detected_foods = []
        self.stable_results_counter = 0
        self.last_processing_time = 0
        
        # Initialize the model and camera
        self._setup_model()
        self._setup_camera()
        
    def _setup_model(self):
        """Setup the FastSAM model for object detection and segmentation."""
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
        print("Model will detect and segment all objects with enhanced CV classification")
        
        # Setup detection parameters - optimized for better accuracy
        self.iou_threshold = 0.45  # Lower for less aggressive filtering
        self.min_mask_area = 300   # Lower threshold for smaller objects
        
    def _setup_camera(self):
        """Setup the camera for video capture with optimized settings."""
        print(f"Initializing camera {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")
            
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Additional optimizations for stability
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
        
        print("Camera initialized successfully!")
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better detection using computer vision techniques.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Enhanced frame for better object detection
        """
        # Convert to float for processing
        frame_float = frame.astype(np.float32) / 255.0
        
        # 1. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # Merge back
        enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 2. Reduce noise while preserving edges using bilateral filter
        denoised = cv2.bilateralFilter(enhanced_frame, 9, 75, 75)
        
        # 3. Slight sharpening to enhance details
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # 4. Gamma correction for better visibility in different lighting
        gamma = 1.2
        gamma_corrected = np.power(sharpened.astype(np.float32) / 255.0, 1.0/gamma)
        gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
        
        return gamma_corrected
    
    def _results_are_similar(self, current_results: List[Tuple[str, float, int, np.ndarray]], 
                           previous_results: List[Tuple[str, float, int, np.ndarray]], 
                           similarity_threshold: float = 0.8) -> bool:
        """
        Check if current detection results are similar to previous ones for stability.
        
        Args:
            current_results: Current detection results
            previous_results: Previous detection results
            similarity_threshold: Threshold for considering results similar
            
        Returns:
            bool: True if results are similar enough
        """
        if not previous_results and not current_results:
            return True
        
        if len(current_results) != len(previous_results):
            return False
        
        if not current_results:
            return True
            
        # Compare object names and approximate positions
        for i, (curr_name, curr_conf, curr_cal, curr_mask) in enumerate(current_results):
            if i >= len(previous_results):
                return False
                
            prev_name, prev_conf, prev_cal, prev_mask = previous_results[i]
            
            # Check if names are similar (allowing for some variation)
            if curr_name != prev_name:
                return False
                
            # Check if confidence is reasonably stable
            if abs(curr_conf - prev_conf) > 0.3:
                return False
                
        return True
        
    def _classify_masked_region(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[str, float]:
        """
        Classify a masked region using color, shape, and texture analysis.
        
        Args:
            frame: Original frame
            mask: Binary mask of the region
            
        Returns:
            Tuple[str, float]: (object_name, food_confidence)
        """
        # Extract the masked region
        mask_bool = mask > 0.5
        if not np.any(mask_bool):
            return "unknown", 0.0
        
        # Resize mask to frame size if needed
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask > 0.5
        
        # Extract masked region
        masked_region = frame.copy()
        masked_region[~mask_bool] = 0
        
        # Color analysis
        food_color_confidence = self._analyze_food_colors(masked_region, mask_bool)
        
        # Shape analysis
        food_shape_confidence = self._analyze_food_shapes(mask_bool)
        
        # Texture analysis
        texture_confidence = self._analyze_texture(masked_region, mask_bool)
        
        # Size analysis
        area = np.sum(mask_bool)
        size_confidence = min(1.0, area / 10000)  # Normalize by typical food size
        
        # Combine confidences with weights
        overall_confidence = (food_color_confidence * 0.4 + 
                             food_shape_confidence * 0.25 + 
                             texture_confidence * 0.2 +
                             size_confidence * 0.15)
        
        # Determine object name based on analysis
        if overall_confidence > 0.7:
            object_name = self._determine_food_type(masked_region, mask_bool)
        elif overall_confidence > 0.4:
            object_name = "possible_food"
        else:
            object_name = f"object_{np.random.randint(1, 100)}"
        
        return object_name, overall_confidence

    def _analyze_food_colors(self, masked_region: np.ndarray, mask_bool: np.ndarray) -> float:
        """Analyze colors to determine if it's likely food."""
        if not np.any(mask_bool):
            return 0.0
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)
        
        # Define food color ranges
        food_color_ranges = [
            # Fruits (reds, oranges, yellows, greens)
            ([0, 50, 50], [10, 255, 255]),    # Red (apples, tomatoes)
            ([8, 50, 50], [25, 255, 255]),    # Orange/Yellow (oranges, bananas)
            ([35, 50, 50], [85, 255, 255]),   # Green (vegetables, fruits)
            # Cooked food (browns)
            ([10, 50, 50], [20, 255, 200]),   # Brown (bread, meat)
            # Vegetables
            ([25, 50, 50], [35, 255, 255]),   # Yellow-green
            # Purple foods
            ([120, 50, 50], [150, 255, 255]), # Purple (eggplant, grapes)
        ]
        
        total_food_pixels = 0
        total_pixels = np.sum(mask_bool)
        
        for lower, upper in food_color_ranges:
            color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            # Fix type issue: ensure both operands are uint8
            mask_uint8 = (mask_bool.astype(np.uint8) * 255)
            food_pixels = np.sum(cv2.bitwise_and(color_mask, mask_uint8))
            total_food_pixels += food_pixels
        
        return min(1.0, total_food_pixels / (total_pixels * 255) * 2)

    def _analyze_food_shapes(self, mask_bool: np.ndarray) -> float:
        """Analyze shape characteristics for food likelihood."""
        # Find contours
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Calculate shape features
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        
        # Food items often have these characteristics
        shape_score = 0.0
        
        # Circular/oval shapes (fruits, plates)
        if 0.3 <= circularity <= 1.0:
            shape_score += 0.4
        
        # Reasonable aspect ratios
        if 0.3 <= aspect_ratio <= 3.0:
            shape_score += 0.3
        
        # Good solidity (not too many holes/concavities)
        if solidity > 0.7:
            shape_score += 0.3
        
        return min(1.0, shape_score)

    def _analyze_texture(self, masked_region: np.ndarray, mask_bool: np.ndarray) -> float:
        """Analyze texture features for food classification."""
        if not np.any(mask_bool):
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(masked_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features using Local Binary Pattern approach
        # Simplified version - calculate standard deviation and mean
        masked_gray = gray[mask_bool]
        
        if len(masked_gray) < 10:
            return 0.0
        
        # Calculate texture measures
        mean_intensity = float(masked_gray.mean())
        std_intensity = float(masked_gray.std())
        
        # Food textures typically have moderate variation
        # Normalize to 0-1 range
        texture_score = 0.0
        
        # Good contrast (not too flat, not too noisy)
        if 20 < std_intensity < 80:
            texture_score += 0.5
        
        # Reasonable brightness
        if 30 < mean_intensity < 200:
            texture_score += 0.5
        
        return texture_score

    def _determine_food_type(self, masked_region: np.ndarray, mask_bool: np.ndarray) -> str:
        """Determine specific food type based on visual analysis."""
        if not np.any(mask_bool):
            return "unknown_food"
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)
        
        # Analyze dominant colors in the masked region
        masked_hsv = hsv[mask_bool]
        if len(masked_hsv) == 0:
            return "unknown_food"
        
        # Calculate average hue, saturation, and value
        avg_hue = float(masked_hsv[:, 0].mean())
        avg_saturation = float(masked_hsv[:, 1].mean())
        avg_value = float(masked_hsv[:, 2].mean())
        
        # Classify based on color characteristics
        if avg_saturation < 50:  # Low saturation - likely processed/cooked food
            if avg_value > 150:
                return "white_food"  # Rice, bread, etc.
            elif avg_value < 80:
                return "dark_food"   # Meat, dark bread
            else:
                return "cooked_food"
        
        # High saturation - likely fresh produce
        if 0 <= avg_hue <= 15 or 165 <= avg_hue <= 180:
            return "red_fruit"    # Apple, tomato, strawberry
        elif 15 < avg_hue <= 30:
            return "orange_food"  # Orange, carrot, pumpkin
        elif 30 < avg_hue <= 60:
            return "yellow_food"  # Banana, corn, lemon
        elif 60 < avg_hue <= 120:
            return "green_vegetable"  # Lettuce, broccoli, cucumber
        elif 120 < avg_hue <= 150:
            return "purple_food"  # Eggplant, grapes
        else:
            return "colorful_food"
        
    def _segment_food_items(self, frame: np.ndarray) -> List[Tuple[str, float, int, np.ndarray]]:
        """
        Segment all objects from camera frame using FastSAM with enhanced preprocessing.
        
        Args:
            frame (np.ndarray): Camera frame in BGR format
            
        Returns:
            List[Tuple[str, float, int, np.ndarray]]: List of (object_name, confidence, calories, mask)
        """
        if self.model is None:
            raise RuntimeError("FastSAM model is not initialized")
        
        try:
            # Preprocess frame for better detection
            enhanced_frame = self._preprocess_frame(frame)
            
            # Run FastSAM segmentation with optimized parameters
            results = self.model(
                enhanced_frame,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                retina_masks=True,
                imgsz=640,  # Higher resolution for better accuracy
                conf=0.60,   # Lower confidence to catch more objects
                iou=0.45,    # Lower IoU threshold
                max_det=20,  # Allow more detections
                augment=True, # Test-time augmentation
                # Remove texts parameter to detect everything
                texts="glasses"
            )
            
            detected_objects = []
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    print(f"Found {len(masks)} masks from FastSAM")
                    
                    # Get confidence scores if available
                    confidences = None
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, mask in enumerate(masks):
                        mask_area = np.sum(mask > 0.5)
                        if mask_area < self.min_mask_area:
                            continue
                        
                        # Use actual confidence if available
                        base_confidence = confidences[i] if confidences is not None and i < len(confidences) else 0.8
                        
                        # Classify the masked region using computer vision
                        object_name, food_confidence = self._classify_masked_region(frame, mask)
                        
                        # Combine FastSAM confidence with CV classification confidence
                        final_confidence = (base_confidence * 0.6 + food_confidence * 0.4)
                        
                        # Calculate calories based on classification
                        calories = 0
                        if food_confidence > 0.4:  # Only if likely food
                            calories = self.calorie_estimator.estimate_calories_from_mask(
                                object_name, mask, frame.shape
                            )
                        
                        detected_objects.append((object_name, final_confidence, calories, mask))
                        
                        print(f"Detected: {object_name} (FastSAM: {base_confidence:.2f}, CV: {food_confidence:.2f}, Final: {final_confidence:.2f})")
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in object segmentation: {e}")
            return []
    
    def _is_likely_food(self, object_name: str) -> bool:
        """
        Check if an object name suggests it's likely food.
        
        Args:
            object_name (str): Name of the detected object
            
        Returns:
            bool: True if the object is likely food
        """
        # Common food-related terms that might appear in class names
        food_keywords = [
            'food', 'eat', 'fruit', 'vegetable', 'meat', 'bread', 'cake', 'pizza',
            'sandwich', 'apple', 'banana', 'orange', 'carrot', 'tomato', 'lettuce',
            'burger', 'hotdog', 'donut', 'cookie', 'cheese', 'milk', 'egg',
            'chicken', 'beef', 'fish', 'rice', 'pasta', 'salad', 'soup'
        ]
        
        object_name_lower = object_name.lower()
        return any(keyword in object_name_lower for keyword in food_keywords)
    
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
            print(f"Drawing {len(detected_foods)} detected objects")
            # Create a separate mask overlay for better control
            mask_overlay = np.zeros_like(frame)
            
            for i, (object_name, confidence, calories, mask) in enumerate(detected_foods):
                print(f"Drawing object {i+1}: {object_name}, mask shape: {mask.shape}")
                
                # Generate a unique, bright color for each object
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
                
                # Ensure mask has the right shape
                if len(mask.shape) == 2:
                    # Resize mask to match frame if needed
                    if mask.shape != (height, width):
                        mask = cv2.resize(mask, (width, height))
                    
                    # Apply colored mask with higher alpha for visibility
                    mask_bool = mask > 0.3  # Lower threshold for better visibility
                    mask_overlay[mask_bool] = color
                    
                    # Draw thick contour outline for better visibility
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(frame, contours, -1, color, 3)  # Thicker contour
                    
                    # Add object label near the center of the mask
                    if np.any(mask_bool):
                        # Find centroid of the mask
                        moments = cv2.moments(mask_uint8)
                        if moments["m00"] != 0:
                            cx = int(moments["m10"] / moments["m00"])
                            cy = int(moments["m01"] / moments["m00"])
                            
                            # Draw object name with background
                            label = f"{object_name.replace('_', ' ').title()}"
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
            
            # Blend mask overlay with original frame for better visibility
            alpha = 0.6  # Higher transparency for better mask visibility
            if np.any(mask_overlay):
                cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                print("No mask overlay to blend")
        else:
            print("No detected objects to draw")
        
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
            cv2.putText(frame, "Status: Scanning for objects...", (20, y_offset), 
                       font, 0.7, (255, 165, 0), 2)  # Orange color
            cv2.putText(frame, "Point camera at objects to detect", (20, y_offset + 25), 
                       font, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Found {len(detected_foods)} object(s):", (20, y_offset), 
                       font, 0.6, (0, 255, 0), 2)  # Green color
            
            y_offset += 30
            total_calories = 0
            food_count = 0
            
            for i, (object_name, confidence, calories, _) in enumerate(detected_foods):
                display_name = object_name.replace('_', ' ').title()
                if calories > 0:
                    text = f"{i+1}. {display_name}: {calories} cal ({confidence:.1%}) [FOOD]"
                    food_count += 1
                else:
                    text = f"{i+1}. {display_name} ({confidence:.1%})"
                cv2.putText(frame, text, (20, y_offset), 
                           font, 0.5, (255, 255, 255), 1)
                y_offset += 20
                total_calories += calories
            
            # Show total calories if any food was detected
            if total_calories > 0:
                cv2.putText(frame, f"Total Food Calories: {total_calories} ({food_count} food items)", (20, y_offset), 
                           font, 0.6, (0, 255, 255), 2)  # Cyan color
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit", (width - 200, height - 20), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Show detection info
        cv2.putText(frame, "FastSAM Object Detection", (width - 250, 30), 
                   font, 0.6, (255, 255, 0), 2)
        
        return frame
        
    def _add_performance_info(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """
        Add performance information to the display frame.
        
        Args:
            frame: Current display frame
            current_time: Current timestamp
            
        Returns:
            Frame with performance info overlay
        """
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Performance info box (top right)
        info_x = width - 280
        info_y = 50
        info_height = 120
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x, info_y), (width - 10, info_y + info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add performance text
        y_offset = info_y + 25
        cv2.putText(frame, "Performance Info:", (info_x + 10, y_offset), 
                   font, 0.5, (255, 255, 0), 1)
        
        y_offset += 20
        cv2.putText(frame, f"Frame: {self.frame_count}", (info_x + 10, y_offset), 
                   font, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(frame, f"Skip Rate: 1/{self.process_every_n_frames}", (info_x + 10, y_offset), 
                   font, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(frame, f"Proc Time: {self.last_processing_time:.3f}s", (info_x + 10, y_offset), 
                   font, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        cv2.putText(frame, f"Stable: {self.stable_results_counter}", (info_x + 10, y_offset), 
                   font, 0.4, (255, 255, 255), 1)
        
        y_offset += 15
        next_process = self.process_every_n_frames - (self.frame_count % self.process_every_n_frames)
        cv2.putText(frame, f"Next Proc: {next_process}", (info_x + 10, y_offset), 
                   font, 0.4, (0, 255, 255), 1)
        
        return frame
        
    def run(self):
        """Run the FastSAM object detection app with frame skipping for improved performance."""
        print("Starting FastSAM Object Detection App...")
        print("The app will detect and segment all objects in the scene!")
        print("Food items will be highlighted with calorie information.")
        print(f"Processing every {self.process_every_n_frames} frames for better performance")
        print("Press 'q' to quit the application.")
        
        self.running = True
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.cap.read() # type: ignore
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Clear camera buffer if processing is taking too long
                if self.last_processing_time > 0.5:  # If processing takes more than 500ms
                    # Skip a few frames to clear buffer
                    for _ in range(2):
                        self.cap.read()  # type: ignore
                
                self.frame_count += 1
                current_time = time.time()
                
                # Only process detection every N frames
                should_process = (self.frame_count % self.process_every_n_frames == 0)
                
                if should_process:
                    processing_start = time.time()
                    
                    # Segment food items using FastSAM
                    detected_foods = self._segment_food_items(frame)
                    
                    processing_time = time.time() - processing_start
                    self.last_processing_time = processing_time
                    
                    # Check if results are stable
                    if self._results_are_similar(detected_foods, self.last_detected_foods):
                        self.stable_results_counter += 1
                    else:
                        self.stable_results_counter = 0
                        
                    # Only update results if they're stable or this is a new detection
                    if self.stable_results_counter >= self.stabilization_frames or not self.last_detected_foods:
                        self.last_detected_foods = detected_foods
                    
                    print(f"Processing time: {processing_time:.3f}s, Stable frames: {self.stable_results_counter}")
                
                # Always draw overlay with the most recent stable results
                display_frame = self._draw_overlay(frame, self.last_detected_foods)
                
                # Add performance info to display
                display_frame = self._add_performance_info(display_frame, current_time)
                
                # Display the frame
                cv2.imshow('FastSAM Object Detection', display_frame)
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update FPS every 30 frames
                    fps = fps_counter / (time.time() - fps_start_time)
                    print(f"Display FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
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
    import sys
    
    try:
        # Use moderate confidence threshold for object detection
        print(f"Starting FastSAM Object Detection App...")
        print("Detecting all objects with enhanced CV-based food identification and calorie estimation")
        print("New Features:")
        print("- Advanced frame preprocessing (CLAHE, denoising, sharpening)")
        print("- Computer vision-based classification (color, shape, texture analysis)")
        print("- Improved object detection with higher resolution processing")
        print("- Intelligent food type determination")
        print("Performance optimizations:")
        print("- Processing every 5th frame for better speed")
        print("- Results stabilization over 3 frames")
        print("- Real-time performance monitoring")
            
        app = FoodRecognitionApp(
            camera_index=0, 
            confidence_threshold=0.4,
            show_all_masks=True,  # Always show all objects now
            process_every_n_frames=5,  # Process every 5th frame
            stabilization_frames=3  # Require 3 stable frames before updating
        )
        app.run()
    except Exception as e:
        print(f"Failed to start FastSAM object detection application: {e}")
        print("Make sure your camera is available and not being used by another application.")
        print("Also ensure ultralytics is installed: pip install ultralytics")
        print("FastSAM model will be downloaded automatically on first run.")
        print("\nUsage:")
        print("  python food_recognition_app.py              # Detect all objects with optimizations")


if __name__ == "__main__":
    main()
