#!/usr/bin/env python3
"""
FastSAM Food Recognition Demo with Enhanced Mask Visualization

This is a demo script to test the enhanced segmentation mask display features.
"""

from food_recognition_app import FoodRecognitionApp

def main():
    print("=" * 60)
    print("FastSAM Food Recognition - Enhanced Mask Visualization Demo")
    print("=" * 60)
    print()
    print("Features:")
    print("✓ Real-time segmentation with FastSAM")
    print("✓ Colored mask overlays for each detected object")
    print("✓ Food classification with confidence scores")
    print("✓ Calorie estimation based on segment size")
    print("✓ Texture + Color + Shape analysis")
    print()
    print("Controls:")
    print("  'q' - Quit the application")
    print("  'm' - Toggle between FOOD ONLY and ALL OBJECTS modes")
    print()
    print("Mask Visualization:")
    print("• Each detected object gets a unique bright color")
    print("• Thick contour outlines for better visibility")
    print("• Food labels displayed at segment centers")
    print("• Semi-transparent overlay preserves original image")
    print()
    
    try:
        # Start with food-only mode
        app = FoodRecognitionApp(
            camera_index=0, 
            confidence_threshold=0.4,
            show_all_masks=False  # Start in food mode
        )
        
        print("Starting camera... Point at food items to see segmentation!")
        print("Press 'm' during runtime to see ALL segmented objects")
        print()
        
        app.run()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is not being used by another app")
        print("2. Check that ultralytics is installed: pip install ultralytics")
        print("3. Ensure you have a working camera connected")

if __name__ == "__main__":
    main()
