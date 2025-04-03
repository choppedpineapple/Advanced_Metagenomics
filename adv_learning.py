"""
PYTHON CONCEPTS BUILDER: FROM BASICS TO STRUCTURED CODE
Follow in order, read comments, and experiment with modifications!
"""

# ====================
# SECTION 1: FUNCTIONAL BUILDING BLOCKS
# Learn to decompose problems into reusable functions
# ====================

def basic_math_operations(a, b):
    """Demonstrates basic function structure and returns"""
    addition = a + b
    subtraction = a - b
    return {  # Returning dictionary for structured data
        "add": addition,
        "sub": subtraction,
    }

# Why functions? 
# 1. Reusability: Encapsulate logic for repeated use
# 2. Readability: Named operations clarify intent
# 3. Testability: Isolated functionality for easier verification

result = basic_math_operations(5, 3)
print(f"Basic Operations Result: {result}")

# Advanced Thinking: Notice we return a dictionary. Why not just print?
# Answer: Separation of concerns. The function calculates, handling output 
# elsewhere makes it more flexible for future changes.

# ====================
# SECTION 2: ERROR HANDLING & VALIDATION
# Graceful failure handling and input protection
# ====================

def validated_math_operations(a, b):
    """Adds type checking and error handling"""
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise TypeError("Both inputs must be numbers")  # Failing fast
    
    try:
        division = a / b
    except ZeroDivisionError:
        division = "Undefined (division by zero)"
    
    return {
        "add": a + b,
        "sub": a - b,
        "div": division
    }

# Why validate?
# 1. Defensive programming prevents silent errors
# 2. Clear error messages speed up debugging
# 3. Type checking ensures expected usage

try:
    print(validated_math_operations(10, 0))
except TypeError as e:
    print(f"Validation Error: {e}")

# Thinking Exercise: What if we wanted to log errors instead of printing?
# That's where the separation of concerns becomes valuable.

# ====================
# SECTION 3: CLASS-BASED ABSTRACTION
# Creating logical entities with state and behavior
# ====================

class MathProcessor:
    """Encapsulates math operations with state tracking"""
    
    def __init__(self, precision=2):
        self.precision = precision  # Instance state
        self.history = []  # Maintaining operation history
    
    def execute(self, a, b, operation):
        """Generic execution with history tracking"""
        import decimal  # Using precise decimal arithmetic
        result = None
        
        # Strategy pattern: Delegate to specific methods
        if operation == 'add':
            result = self._add(a, b)
        elif operation == 'sub':
            result = self._subtract(a, b)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Format result consistently
        formatted = round(result, self.precision) if isinstance(result, float) else result
        self.history.append({
            "operation": operation,
            "result": formatted
        })
        return formatted
    
    def _add(self, a, b):
        """Internal implementation detail"""
        return a + b
    
    def _subtract(self, a, b):
        return a - b

# Why classes?
# 1. Encapsulation: Group related data and behavior
# 2. State management: Maintain history between operations
# 3. Implementation hiding: Users interact with execute(), not internal methods

processor = MathProcessor(precision=3)
print(f"Class-Based Result: {processor.execute(5.1234, 2.3456, 'add')}")
print(f"Operation History: {processor.history}")

# Design Insight: The public execute() method acts as an API facade. This
# allows changing internal implementations without affecting users.

# ====================
# SECTION 4: DESIGN PATTERNS IN ACTION
# Implementing flexible architecture patterns
# ====================

from abc import ABC, abstractmethod

class OperationStrategy(ABC):
    """Strategy pattern interface for math operations"""
    @abstractmethod
    def execute(self, a, b):
        pass

class AdditionStrategy(OperationStrategy):
    def execute(self, a, b):
        return a + b

class DivisionStrategy(OperationStrategy):
    def execute(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

class AdvancedMathProcessor:
    """Uses strategy pattern for extensible operations"""
    def __init__(self):
        self.strategies = {
            'add': AdditionStrategy(),
            'div': DivisionStrategy()
        }
    
    def execute(self, a, b, operation):
        strategy = self.strategies.get(operation)
        if not strategy:
            raise ValueError(f"Unsupported operation: {operation}")
        return strategy.execute(a, b)

# Why patterns?
# 1. Open/closed principle: Add new operations without modifying existing code
# 2. Polymorphism: Uniform interface for diverse operations
# 3. Testability: Each strategy can be tested in isolation

advanced_processor = AdvancedMathProcessor()
print(f"Strategy Result: {advanced_processor.execute(10, 2, 'div')}")

# Architectural Thinking: Notice how new operations just need a strategy class.
# This structure makes the system more maintainable as it grows.

# ====================
# SECTION 5: MODULARIZATION & PACKAGES
# Organizing code for large-scale applications
# ====================

# Imagine this in separate files:
# File: operations/strategies.py
class MultiplicationStrategy(OperationStrategy):
    def execute(self, a, b):
        return a * b

# File: math_processor.py
from operations.strategies import MultiplicationStrategy

class ExtendedMathProcessor(AdvancedMathProcessor):
    def __init__(self):
        super().__init__()
        self.strategies['mul'] = MultiplicationStrategy()

# File: main.py
# from math_processor import ExtendedMathProcessor

# Why modules?
# 1. Separation of concerns
# 2. Collaborative development
# 3. Namespace management
# 4. Reusable components

# ====================
# SECTION 6: TESTING & QUALITY
# Ensuring reliability with automated tests
# ====================

import unittest

class TestMathOperations(unittest.TestCase):
    def test_addition(self):
        proc = AdvancedMathProcessor()
        self.assertEqual(proc.execute(2, 3, 'add'), 5)
    
    def test_division_error(self):
        proc = AdvancedMathProcessor()
        with self.assertRaises(ValueError):
            proc.execute(5, 0, 'div')

# Run tests with:
# if __name__ == "__main__":
#     unittest.main()

# Testing Philosophy:
# 1. Write tests before (TDD) or alongside code
# 2. Tests document system behavior
# 3. Enable safe refactoring

# ====================
# SECTION 7: PUTTING IT ALL TOGETHER
# Final implementation with best practices
# ====================

"""
Full implementation would include:
1. Modular code structure
2. Comprehensive error handling
3. Unit tests
4. Documentation
5. Logging
6. Configuration management

Exercise: Extend this system with:
1. New operation (exponentiation)
2. JSON history export
3. Configuration file support
4. CLI interface
"""

# Final Wisdom:
# 1. Code is read more than written - prioritize clarity
# 2. Design for change - requirements evolve
# 3. Invest in testing - saves time long-term
# 4. Learn patterns, but apply judiciously - not every problem needs a pattern
