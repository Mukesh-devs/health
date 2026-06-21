"""
Health LLM Application - Entry Point
"""
import os
from dotenv import load_dotenv
from healthapp import create_app

# Load environment variables from .env file
load_dotenv()

# Create app instance using factory pattern
config_name = os.getenv('FLASK_ENV', 'default')
app = create_app(config_name)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
