#!/usr/bin/env python3
"""
Test script to verify mask visualization works without camera
"""

import cv2
import numpy as np

def test_mask_visualization():
    """Test the mask overlay functionality"""
    # Create a test image
    height, width = 480, 640
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame.fill(50)  # Dark gray background
    
    # Add some content to the frame
    cv2.rectangle(frame, (100, 100), (200, 200), (100, 100, 100), -1)
    cv2.rectangle(frame, (300, 150), (500, 350), (80, 80, 80), -1)
    
    # Create test masks
    mask1 = np.zeros((height, width), dtype=np.float32)
    mask1[50:150, 50:150] = 1.0
    
    mask2 = np.zeros((height, width), dtype=np.float32)
    mask2[200:300, 250:350] = 1.0
    
    mask3 = np.zeros((height, width), dtype=np.float32)
    # Create a circular mask
    center = (400, 200)
    radius = 60
    y, x = np.ogrid[:height, :width]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    mask3[mask_circle] = 1.0
    
    # Test masks and info
    detected_objects = [
        ("test_rectangle", 0.9, 150, mask1),
        ("test_square", 0.8, 0, mask2),
        ("test_circle", 0.95, 200, mask3)
    ]
    
    # Draw masks with colors
    mask_overlay = np.zeros_like(frame)
    colors = [
        [0, 255, 0],      # Bright green
        [255, 0, 0],      # Bright blue (BGR)
        [0, 0, 255],      # Bright red
    ]
    
    for i, (object_name, confidence, calories, mask) in enumerate(detected_objects):
        color = colors[i % len(colors)]
        mask_bool = mask > 0.5
        mask_overlay[mask_bool] = color
        
        # Draw contours
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 3)
        
        # Add labels
        if np.any(mask_bool):
            moments = cv2.moments(mask_uint8)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                
                label = object_name.replace('_', ' ').title()
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, label, (cx - 50, cy), font, 0.6, color, 2)
    
    # Blend mask overlay
    alpha = 0.6
    if np.any(mask_overlay):
        cv2.addWeighted(mask_overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add info text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    cv2.putText(frame, "Mask Visualization Test", (20, y_offset), font, 0.8, (255, 255, 255), 2)
    y_offset += 40
    
    for i, (object_name, confidence, calories, _) in enumerate(detected_objects):
        display_name = object_name.replace('_', ' ').title()
        if calories > 0:
            text = f"{i+1}. {display_name}: {calories} cal ({confidence:.1%}) [FOOD]"
        else:
            text = f"{i+1}. {display_name} ({confidence:.1%})"
        cv2.putText(frame, text, (20, y_offset), font, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    cv2.putText(frame, "Press any key to close", (20, height - 20), font, 0.5, (255, 255, 255), 1)
    
    # Display the test
    cv2.imshow('Mask Visualization Test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Test completed - masks should have been visible with colored overlays and contours")

if __name__ == "__main__":
    test_mask_visualization()
