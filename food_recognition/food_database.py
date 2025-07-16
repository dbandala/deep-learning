"""
Food Database for Food Recognition App

Contains information about various food items and their properties.
"""

from typing import Dict, List, Optional


class FoodDatabase:
    """Database of food items with nutritional information."""
    
    def __init__(self):
        """Initialize the food database with common food items."""
        self.foods = {
            0: {"name": "apple", "calories_per_100g": 52, "typical_serving": 150},
            1: {"name": "banana", "calories_per_100g": 89, "typical_serving": 120},
            2: {"name": "orange", "calories_per_100g": 47, "typical_serving": 130},
            3: {"name": "bread_slice", "calories_per_100g": 265, "typical_serving": 30},
            4: {"name": "pizza_slice", "calories_per_100g": 266, "typical_serving": 100},
            5: {"name": "hamburger", "calories_per_100g": 295, "typical_serving": 150},
            6: {"name": "sandwich", "calories_per_100g": 250, "typical_serving": 120},
            7: {"name": "salad", "calories_per_100g": 20, "typical_serving": 200},
            8: {"name": "rice", "calories_per_100g": 130, "typical_serving": 150},
            9: {"name": "pasta", "calories_per_100g": 131, "typical_serving": 150},
            10: {"name": "chicken_breast", "calories_per_100g": 165, "typical_serving": 120},
            11: {"name": "french_fries", "calories_per_100g": 365, "typical_serving": 85},
            12: {"name": "donut", "calories_per_100g": 452, "typical_serving": 60},
            13: {"name": "cookie", "calories_per_100g": 502, "typical_serving": 25},
            14: {"name": "cake", "calories_per_100g": 257, "typical_serving": 80},
            15: {"name": "ice_cream", "calories_per_100g": 207, "typical_serving": 65},
            16: {"name": "yogurt", "calories_per_100g": 59, "typical_serving": 150},
            17: {"name": "cheese", "calories_per_100g": 113, "typical_serving": 30},
            18: {"name": "egg", "calories_per_100g": 155, "typical_serving": 50},
            19: {"name": "milk", "calories_per_100g": 42, "typical_serving": 250},
            20: {"name": "cereal", "calories_per_100g": 379, "typical_serving": 40},
            21: {"name": "soup", "calories_per_100g": 84, "typical_serving": 250},
            22: {"name": "fish", "calories_per_100g": 206, "typical_serving": 100},
            23: {"name": "steak", "calories_per_100g": 271, "typical_serving": 150},
            24: {"name": "hotdog", "calories_per_100g": 290, "typical_serving": 45},
        }
        
    def get_food_by_index(self, index: int) -> str:
        """
        Get food name by index.
        
        Args:
            index (int): Food index
            
        Returns:
            str: Food name or "unknown" if index not found
        """
        if index in self.foods:
            return self.foods[index]["name"]
        return "unknown"
        
    def get_food_info(self, food_name: str) -> Optional[Dict]:
        """
        Get food information by name.
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            Optional[Dict]: Food information dict or None if not found
        """
        for food_info in self.foods.values():
            if food_info["name"].lower() == food_name.lower():
                return food_info
        return None
        
    def get_calories_per_100g(self, food_name: str) -> int:
        """
        Get calories per 100g for a food item.
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            int: Calories per 100g, or 0 if not found
        """
        food_info = self.get_food_info(food_name)
        if food_info:
            return food_info["calories_per_100g"]
        return 0
        
    def get_typical_serving(self, food_name: str) -> int:
        """
        Get typical serving size in grams for a food item.
        
        Args:
            food_name (str): Name of the food
            
        Returns:
            int: Typical serving size in grams, or 100 if not found
        """
        food_info = self.get_food_info(food_name)
        if food_info:
            return food_info["typical_serving"]
        return 100
        
    def get_all_foods(self) -> List[str]:
        """
        Get list of all food names in the database.
        
        Returns:
            List[str]: List of all food names
        """
        return [food_info["name"] for food_info in self.foods.values()]
        
    def get_num_classes(self) -> int:
        """
        Get the number of food classes in the database.
        
        Returns:
            int: Number of food classes
        """
        return len(self.foods)
        
    def add_food(self, name: str, calories_per_100g: int, typical_serving: int) -> int:
        """
        Add a new food item to the database.
        
        Args:
            name (str): Food name
            calories_per_100g (int): Calories per 100g
            typical_serving (int): Typical serving size in grams
            
        Returns:
            int: Index of the newly added food item
        """
        new_index = max(self.foods.keys()) + 1 if self.foods else 0
        self.foods[new_index] = {
            "name": name,
            "calories_per_100g": calories_per_100g,
            "typical_serving": typical_serving
        }
        return new_index
