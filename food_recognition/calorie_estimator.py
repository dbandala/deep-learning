"""
Calorie Estimator for Food Recognition App

Estimates calories for detected food items based on confidence and serving sizes.
"""

import math
from typing import Optional
from food_database import FoodDatabase


class CalorieEstimator:
    """Estimates calories for food items based on detection confidence and serving sizes."""
    
    def __init__(self):
        """Initialize the calorie estimator."""
        self.food_db = FoodDatabase()
        
    def estimate_calories(self, food_name: str, confidence: float, 
                         serving_factor: float = 1.0) -> int:
        """
        Estimate calories for a detected food item.
        
        Args:
            food_name (str): Name of the detected food
            confidence (float): Detection confidence (0.0 to 1.0)
            serving_factor (float): Factor to adjust serving size (default: 1.0)
            
        Returns:
            int: Estimated calories
        """
        if food_name == "unknown" or food_name == "No food detected":
            return 0
            
        # Get food information from database
        calories_per_100g = self.food_db.get_calories_per_100g(food_name)
        typical_serving = self.food_db.get_typical_serving(food_name)
        
        if calories_per_100g == 0:
            return 0
            
        # Calculate base calories for typical serving
        base_calories = (calories_per_100g * typical_serving) / 100
        
        # Adjust based on confidence (lower confidence might mean partial food item)
        confidence_factor = self._get_confidence_factor(confidence)
        
        # Apply serving factor
        adjusted_calories = base_calories * confidence_factor * serving_factor
        
        return max(1, int(round(adjusted_calories)))
        
    def _get_confidence_factor(self, confidence: float) -> float:
        """
        Calculate a factor based on detection confidence.
        
        Lower confidence might indicate:
        - Partial food item visible
        - Unclear/blurry image
        - Food item partially eaten
        
        Args:
            confidence (float): Detection confidence
            
        Returns:
            float: Confidence factor for calorie adjustment
        """
        if confidence >= 0.8:
            return 1.0  # Full serving
        elif confidence >= 0.6:
            return 0.8  # Most of the serving
        elif confidence >= 0.4:
            return 0.6  # Partial serving
        elif confidence >= 0.3:
            return 0.4  # Small portion
        else:
            return 0.2  # Very small portion
            
    def estimate_calories_with_size(self, food_name: str, confidence: float,
                                   estimated_weight: Optional[float] = None) -> int:
        """
        Estimate calories with a specific weight estimate.
        
        Args:
            food_name (str): Name of the detected food
            confidence (float): Detection confidence
            estimated_weight (Optional[float]): Estimated weight in grams
            
        Returns:
            int: Estimated calories
        """
        if food_name == "unknown" or food_name == "No food detected":
            return 0
            
        calories_per_100g = self.food_db.get_calories_per_100g(food_name)
        
        if calories_per_100g == 0:
            return 0
            
        if estimated_weight is None:
            # Fall back to typical serving with confidence adjustment
            return self.estimate_calories(food_name, confidence)
            
        # Calculate calories based on estimated weight
        calories = (calories_per_100g * estimated_weight) / 100
        
        # Apply confidence factor
        confidence_factor = self._get_confidence_factor(confidence)
        adjusted_calories = calories * confidence_factor
        
        return max(1, int(round(adjusted_calories)))
        
    def get_nutrition_info(self, food_name: str, confidence: float) -> dict:
        """
        Get comprehensive nutrition information for a food item.
        
        Args:
            food_name (str): Name of the detected food
            confidence (float): Detection confidence
            
        Returns:
            dict: Dictionary with nutrition information
        """
        estimated_calories = self.estimate_calories(food_name, confidence)
        food_info = self.food_db.get_food_info(food_name)
        
        if not food_info:
            return {
                "food_name": food_name,
                "estimated_calories": 0,
                "confidence": confidence,
                "calories_per_100g": 0,
                "typical_serving_g": 0,
                "confidence_factor": 0
            }
            
        return {
            "food_name": food_name,
            "estimated_calories": estimated_calories,
            "confidence": confidence,
            "calories_per_100g": food_info["calories_per_100g"],
            "typical_serving_g": food_info["typical_serving"],
            "confidence_factor": self._get_confidence_factor(confidence)
        }
