import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the app
try:
    from app import create_workflow
    app = create_workflow()
except ImportError as e:
    print(f"Failed to import app: {e}")
    sys.exit(1)

print("Invoking app...")
try:
    result = app.invoke({
        "messages":[
            {
                "role": "user",
                "content": "Based on the GDP of the United States in the past three years, draw a line chart."
            }
        ]
    })
    print("\n--- Result ---")
    print(result)
except Exception as e:
    print(f"Execution failed: {e}")
