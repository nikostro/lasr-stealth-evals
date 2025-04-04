from typing import Callable, Optional, Dict, Any


class ExampleClass:
    def __init__(self):
        # Instance variables
        self.state = 0
        self.functions: Dict[str, Callable[..., int]] = {}

    def register_functions(self, *funcs: Callable[..., int]) -> None:
        """Register multiple functions at once"""
        for func in funcs:
            # Use the function's name as the key
            self.functions[func.__name__] = func

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Optional[int]:
        """Execute a function by name with the given arguments"""
        if name in self.functions:
            return self.functions[name](self, *args, **kwargs)
        return None


# Example usage:
def increment_state(instance: ExampleClass, amount: int = 1) -> int:
    instance.state += amount
    return instance.state


def decrement_state(instance: ExampleClass, amount: int = 1) -> int:
    instance.state -= amount
    return instance.state


def multiply_state(instance: ExampleClass, factor: int) -> int:
    instance.state *= factor
    return instance.state


# Create instances and register functions
instance1 = ExampleClass()
instance2 = ExampleClass()

# Register functions for each instance
instance1.register_functions(increment_state, multiply_state)
instance2.register_functions(decrement_state)

# Execute the functions
print(f"Instance 1 initial state: {instance1.state}")  # 0
print(f"Instance 2 initial state: {instance2.state}")  # 0

# Execute different functions with different arguments
print(f"Instance 1 increment by 2: {instance1.execute('increment_state', 2)}")  # 2
print(f"Instance 1 multiply by 3: {instance1.execute('multiply_state', 3)}")  # 6
print(f"Instance 2 decrement by 1: {instance2.execute('decrement_state')}")  # -1

print(f"Instance 1 final state: {instance1.state}")  # 6
print(f"Instance 2 final state: {instance2.state}")  # -1
