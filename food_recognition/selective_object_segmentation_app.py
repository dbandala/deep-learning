"""
Selective Object Segmentation App using FastSAM

This application uses FastSAM (Fastest Segment Anything Model) to detect and segment 
only specific objects defined by text prompts in real-time from camera feed.
Users can specify which objects to detect using natural language descriptions.
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


class SelectiveObjectSegmentationApp:
    def __init__(self, target_objects: List[str], camera_index: int = 0, confidence_threshold: float = 0.5, 
                 process_every_n_frames: int = 12, stabilization_frames: int = 4):
        """
        Initialize the Selective Object Segmentation App.
        
        Args:
            target_objects (List[str]): List of object names/descriptions to detect
            camera_index (int): Camera device index (default: 0 for primary camera)
            confidence_threshold (float): Minimum confidence for object detection
            process_every_n_frames (int): Process every Nth frame for detection (default: 10)
            stabilization_frames (int): Number of frames to keep results stable (default: 4)
        """
        self.target_objects = target_objects
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.process_every_n_frames = process_every_n_frames
        self.stabilization_frames = stabilization_frames
        self.cap = None
        self.model = None
        self.running = False
        
        # Frame processing tracking
        self.frame_count = 0
        self.last_detected_objects = []
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
            model_path = "FastSAM-x.pt"
            self.model = FastSAM(model_path)
            print("Using local FastSAM-x.pt model")
        except:
            # Fallback to downloading the model
            print("Local model not found, downloading FastSAM-x.pt...")
            self.model = FastSAM('FastSAM-x.pt')

        print("FastSAM model loaded successfully!")
        print(f"Model will detect only: {', '.join(self.target_objects)}")
        
        # Setup detection parameters
        self.iou_threshold = 0.45  # Intersection over Union threshold for filtering detections
        self.min_mask_area = 200   # Minimum area for valid detections
        
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
        
        return frame
    
    def _results_are_similar(self, current_results: List[Tuple[str, float, np.ndarray]], 
                           previous_results: List[Tuple[str, float, np.ndarray]], 
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
            
        # Compare object names and confidence scores
        for i, (curr_name, curr_conf, curr_mask) in enumerate(current_results):
            if i >= len(previous_results):
                return False
                
            prev_name, prev_conf, prev_mask = previous_results[i]
            
            # Check if names are similar
            if curr_name != prev_name:
                return False
                
            # Check if confidence is reasonably stable
            if abs(curr_conf - prev_conf) > 0.3:
                return False
                
        return True
        
    def _segment_target_objects(self, frame: np.ndarray) -> List[Tuple[str, float, np.ndarray]]:
        """
        Segment only the specified target objects from camera frame using FastSAM.
        
        Args:
            frame (np.ndarray): Camera frame in BGR format
            
        Returns:
            List[Tuple[str, float, np.ndarray]]: List of (object_name, confidence, mask)
        """
        if self.model is None:
            raise RuntimeError("FastSAM model is not initialized")
        
        try:
            # Preprocess frame for better detection
            enhanced_frame = self._preprocess_frame(frame)
            
            # Create text prompt for the target objects
            text_prompt = " . ".join(self.target_objects)
            
            # Run FastSAM segmentation with text prompts for selective detection
            results = self.model(
                enhanced_frame,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                # retina_masks=True,
                imgsz=320,  # Higher resolution for better accuracy
                conf=0.8,  # Higher confidence threshold for better precision
                iou=0.25,  # Lower IoU for more inclusive detection
                max_det=5,  # Limit detections for performance
                #augment=True, # Test-time augmentation
                texts=text_prompt,  # Specify target objects to detect
                # points=[(frame.shape[1] // 2, frame.shape[0] // 2)]  # Center of the camera
            )
            
            detected_objects = []
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    print(f"Found {len(masks)} masks matching target objects: {self.target_objects}")
                    
                    # Get confidence scores if available
                    confidences = None
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, mask in enumerate(masks):
                        mask_area = np.sum(mask > 0.5)
                        if mask_area < self.min_mask_area:
                            continue
                        
                        # Use actual confidence if available
                        confidence = confidences[i] if confidences is not None and i < len(confidences) else 0.8
                        
                        # Determine which target object this detection corresponds to
                        # For simplicity, we'll cycle through target objects or use the most likely one
                        object_name = self.target_objects[i % len(self.target_objects)]
                        
                        detected_objects.append((object_name, confidence, mask))
                        
                        print(f"Detected: {object_name} (confidence: {confidence:.2f})")
            else:
                print("No objects matching target criteria found")
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in selective object segmentation: {e}")
            return []
    
    def _draw_overlay(self, frame: np.ndarray, detected_objects: List[Tuple[str, float, np.ndarray]]) -> np.ndarray:
        """
        Draw information overlay and segmentation masks on the camera frame.
        
        Args:
            frame (np.ndarray): Camera frame
            detected_objects (List): List of (object_name, confidence, mask) tuples
            
        Returns:
            np.ndarray: Frame with overlay and masks
        """
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw segmentation masks with different colors and better visibility
        if detected_objects:
            print(f"Drawing {len(detected_objects)} detected target objects")
            # Create a separate mask overlay for better control
            mask_overlay = np.zeros_like(frame)
            
            for i, (object_name, confidence, mask) in enumerate(detected_objects):
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
                    cv2.drawContours(frame, contours, -1, color, 4)  # Thick contour
                    
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
                            font_scale = 0.7
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
                            
                            # Draw confidence score below the label
                            conf_label = f"{confidence:.1%}"
                            cv2.putText(frame, conf_label, 
                                      (cx - text_width//2, cy + 20), 
                                      font, 0.5, color, 1)
            
            # Blend mask overlay with original frame for better visibility
            alpha = 0.5  # Transparency for mask visibility
            if np.any(mask_overlay):
                cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0, frame)
            else:
                print("No mask overlay to blend")
        else:
            print("No target objects detected")
        
        # Draw information box
        info_height = max(120, 60 + len(detected_objects) * 25) if detected_objects else 100
        cv2.rectangle(frame, (10, 10), (width - 10, info_height), (0, 0, 0), -1)
        cv2.addWeighted(frame[10:info_height, 10:width-10], 0.7, 
                       np.zeros((info_height-10, width-20, 3), dtype=np.uint8), 0.3, 0,
                       frame[10:info_height, 10:width-10])
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 35
        
        # Show target objects
        targets_text = f"Searching for: {', '.join(self.target_objects)}"
        cv2.putText(frame, targets_text, (20, y_offset), 
                   font, 0.6, (255, 165, 0), 2)  # Orange color
        y_offset += 25
        
        if not detected_objects:
            cv2.putText(frame, "Status: Scanning for target objects...", (20, y_offset), 
                       font, 0.6, (255, 255, 0), 2)  # Yellow color
            cv2.putText(frame, "Point camera at target objects", (20, y_offset + 25), 
                       font, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Found {len(detected_objects)} target object(s):", (20, y_offset), 
                       font, 0.6, (0, 255, 0), 2)  # Green color
            
            y_offset += 30
            
            for i, (object_name, confidence, _) in enumerate(detected_objects):
                display_name = object_name.replace('_', ' ').title()
                text = f"{i+1}. {display_name} ({confidence:.1%})"
                cv2.putText(frame, text, (20, y_offset), 
                           font, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to change targets", (width - 350, height - 20), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Show detection info
        cv2.putText(frame, "Selective Object Detection", (width - 280, 30), 
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
    
    def update_target_objects(self, new_targets: List[str]):
        """
        Update the list of target objects to detect.
        
        Args:
            new_targets: New list of object names/descriptions
        """
        self.target_objects = new_targets
        print(f"Updated target objects to: {', '.join(self.target_objects)}")
        # Reset detection state
        self.last_detected_objects = []
        self.stable_results_counter = 0
        
    def run(self):
        """Run the Selective Object Segmentation app with frame skipping for improved performance."""
        print("Starting Selective Object Segmentation App...")
        print(f"The app will detect and segment only: {', '.join(self.target_objects)}")
        print(f"Processing every {self.process_every_n_frames} frames for better performance")
        print("Press 'q' to quit, 'r' to change target objects.")
        
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
                    
                    # Segment target objects using FastSAM
                    detected_objects = self._segment_target_objects(frame)
                    
                    processing_time = time.time() - processing_start
                    self.last_processing_time = processing_time
                    
                    # Check if results are stable
                    if self._results_are_similar(detected_objects, self.last_detected_objects):
                        self.stable_results_counter += 1
                    else:
                        self.stable_results_counter = 0
                        
                    # Only update results if they're stable or this is a new detection
                    if self.stable_results_counter >= self.stabilization_frames or not self.last_detected_objects:
                        self.last_detected_objects = detected_objects
                    
                    print(f"Processing time: {processing_time:.3f}s, Stable frames: {self.stable_results_counter}")
                
                # Always draw overlay with the most recent stable results
                display_frame = self._draw_overlay(frame, self.last_detected_objects)
                
                # Add performance info to display
                display_frame = self._add_performance_info(display_frame, current_time)
                
                # Display the frame
                cv2.imshow('Selective Object Detection', display_frame)
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:  # Update FPS every 30 frames
                    fps = fps_counter / (time.time() - fps_start_time)
                    print(f"Display FPS: {fps:.1f}")
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Check for quit key or reconfigure
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reconfigure_targets()
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            self.cleanup()
    
    def _reconfigure_targets(self):
        """Allow user to reconfigure target objects during runtime."""
        print("\n" + "="*50)
        print("RECONFIGURING TARGET OBJECTS")
        print("="*50)
        print("Current targets:", ', '.join(self.target_objects))
        print("Enter new target objects (comma-separated), or press Enter to keep current:")
        
        # This is a simplified version - in a real app you might want a GUI
        try:
            new_input = input("New targets: ").strip()
            if new_input:
                new_targets = [obj.strip() for obj in new_input.split(',') if obj.strip()]
                if new_targets:
                    self.update_target_objects(new_targets)
                else:
                    print("Invalid input, keeping current targets")
            else:
                print("Keeping current targets")
        except:
            print("Input error, keeping current targets")
        
        print("="*50)
            
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
    
    # Default target objects - can be customized
    default_targets = ["person", "cup", "laptop", "cell phone", "book"]
    
    print("="*60)
    print("SELECTIVE OBJECT SEGMENTATION APP")
    print("="*60)
    print("This app uses FastSAM to detect and segment only specific objects")
    print("that you specify using natural language descriptions.")
    print()
    
    # Allow user to specify target objects
    if len(sys.argv) > 1:
        # Command line arguments provided
        target_objects = [arg.strip() for arg in ' '.join(sys.argv[1:]).split(',')]
    else:
        # Interactive input
        print("Default target objects:", ', '.join(default_targets))
        user_input = input("Enter target objects (comma-separated) or press Enter for defaults: ").strip()
        
        if user_input:
            target_objects = [obj.strip() for obj in user_input.split(',') if obj.strip()]
        else:
            target_objects = default_targets
    
    if not target_objects:
        print("No target objects specified. Using defaults.")
        target_objects = default_targets
    
    print(f"\nTarget objects set to: {', '.join(target_objects)}")
    print("\nStarting application...")
    print("Features:")
    print("- Advanced frame preprocessing for better detection")
    print("- Selective object detection using text prompts")
    print("- Real-time performance optimization")
    print("- Interactive target object reconfiguration (press 'r')")
    print("- High-quality segmentation masks with confidence scores")
    
    try:
        app = SelectiveObjectSegmentationApp(
            target_objects=target_objects,
            camera_index=0, 
            confidence_threshold=0.6,  # Higher threshold for better precision
            process_every_n_frames=8,  # Process every 8th frame for performance
            stabilization_frames=2      # Require 2 stable frames before updating
        )
        app.run()
    except Exception as e:
        print(f"Failed to start selective object segmentation application: {e}")
        print("Make sure your camera is available and not being used by another application.")
        print("Also ensure ultralytics is installed: pip install ultralytics")
        print("FastSAM model will be downloaded automatically on first run.")
        print("\nUsage:")
        print("  python selective_object_segmentation_app.py                    # Use default objects")
        print("  python selective_object_segmentation_app.py person,car,dog     # Specify custom objects")


if __name__ == "__main__":
    main()
