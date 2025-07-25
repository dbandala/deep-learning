"""
Demo script for the Selective Object Segmentation App

This script demonstrates how to use the SelectiveObjectSegmentationApp
with different target objects and configurations.
"""

from selective_object_segmentation_app import SelectiveObjectSegmentationApp
import time


def demo_common_objects():
    """Demo with common household objects."""
    print("Demo 1: Common Household Objects")
    print("="*40)
    
    target_objects = ["laptop", "cup", "phone", "book", "mouse"]
    
    app = SelectiveObjectSegmentationApp(
        target_objects=target_objects,
        confidence_threshold=0.6,
        process_every_n_frames=8,
        stabilization_frames=3
    )
    
    print("Starting detection for common objects...")
    print("Point your camera at: laptop, cup, phone, book, or mouse")
    print("Press 'q' to stop this demo and move to the next one")
    
    app.run()
    app.cleanup()


def demo_outdoor_objects():
    """Demo with outdoor/vehicle objects."""
    print("\nDemo 2: Outdoor/Vehicle Objects")
    print("="*40)
    
    target_objects = ["car", "bicycle", "person", "dog", "tree"]
    
    app = SelectiveObjectSegmentationApp(
        target_objects=target_objects,
        confidence_threshold=0.5,
        process_every_n_frames=10,
        stabilization_frames=4
    )
    
    print("Starting detection for outdoor objects...")
    print("Point your camera at: car, bicycle, person, dog, or tree")
    print("Press 'q' to stop this demo and move to the next one")
    
    app.run()
    app.cleanup()


def demo_food_objects():
    """Demo with food objects."""
    print("\nDemo 3: Food Objects")
    print("="*40)
    
    target_objects = ["apple", "banana", "orange", "sandwich", "pizza"]
    
    app = SelectiveObjectSegmentationApp(
        target_objects=target_objects,
        confidence_threshold=0.4,
        process_every_n_frames=12,
        stabilization_frames=5
    )
    
    print("Starting detection for food objects...")
    print("Point your camera at: apple, banana, orange, sandwich, or pizza")
    print("Press 'q' to stop this demo")
    
    app.run()
    app.cleanup()


def demo_single_object():
    """Demo with just one specific object."""
    print("\nDemo 4: Single Object Detection")
    print("="*40)
    
    target_objects = ["person"]  # Just detect people
    
    app = SelectiveObjectSegmentationApp(
        target_objects=target_objects,
        confidence_threshold=0.7,
        process_every_n_frames=6,
        stabilization_frames=2
    )
    
    print("Starting detection for people only...")
    print("Point your camera at people")
    print("Press 'q' to stop this demo")
    
    app.run()
    app.cleanup()


def interactive_demo():
    """Interactive demo where user can change targets during runtime."""
    print("\nInteractive Demo")
    print("="*40)
    
    # Start with some basic objects
    initial_objects = ["bottle", "chair", "keyboard"]
    
    app = SelectiveObjectSegmentationApp(
        target_objects=initial_objects,
        confidence_threshold=0.5,
        process_every_n_frames=8,
        stabilization_frames=3
    )
    
    print("Starting interactive demo...")
    print("Current targets:", ', '.join(initial_objects))
    print("During runtime:")
    print("- Press 'r' to change target objects")
    print("- Press 'q' to quit")
    print("Try changing targets to test different object types!")
    
    app.run()
    app.cleanup()


def main():
    """Run demonstration of the Selective Object Segmentation App."""
    print("SELECTIVE OBJECT SEGMENTATION - DEMO SUITE")
    print("="*50)
    print("This demo will show different configurations of the app.")
    print("Each demo focuses on different types of objects.")
    print()
    
    demos = [
        ("1", "Common household objects", demo_common_objects),
        ("2", "Outdoor/vehicle objects", demo_outdoor_objects),
        ("3", "Food objects", demo_food_objects),
        ("4", "Single object (person) detection", demo_single_object),
        ("5", "Interactive demo (change targets at runtime)", interactive_demo),
    ]
    
    while True:
        print("\nAvailable demos:")
        for num, desc, _ in demos:
            print(f"  {num}. {desc}")
        print("  q. Quit")
        
        choice = input("\nSelect demo (1-5) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        
        # Find and run the selected demo
        demo_found = False
        for num, desc, demo_func in demos:
            if choice == num:
                try:
                    print(f"\nStarting: {desc}")
                    time.sleep(1)  # Brief pause
                    demo_func()
                    demo_found = True
                    break
                except Exception as e:
                    print(f"Error running demo: {e}")
                    print("Make sure your camera is available.")
        
        if not demo_found:
            print("Invalid choice. Please select 1-5 or 'q'.")
    

if __name__ == "__main__":
    main()
