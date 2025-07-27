"""Sample code files for testing test generation capabilities."""


def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price.
    
    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)
        
    Returns:
        Discounted price
    """
    return price * (1 - discount_percent / 100)


class Calculator:
    """Simple calculator class for testing."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def divide(self, a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def factorial(self, n: int) -> int:
        """Calculate factorial of a number."""
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n <= 1:
            return 1
        return n * self.factorial(n - 1)


def process_user_input(user_input: str) -> dict:
    """Process user input (contains potential security issues)."""
    # Security issue: SQL injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    
    # Security issue: Command injection vulnerability  
    import subprocess
    result = subprocess.run(f"echo {user_input}", shell=True, capture_output=True)
    
    return {
        "query": query,
        "output": result.stdout.decode() if result.stdout else ""
    }


def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user (contains security issues)."""
    # Security issue: Hard-coded credentials
    admin_users = {
        "admin": "password123",
        "root": "admin"
    }
    
    # Security issue: Timing attack vulnerability
    if username in admin_users:
        return admin_users[username] == password
    
    return False


async def fetch_user_data(user_id: int) -> dict:
    """Fetch user data asynchronously."""
    import asyncio
    await asyncio.sleep(0.1)  # Simulate API call
    
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    }