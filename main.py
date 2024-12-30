#!/usr/bin/env/python3

"""
    main.py

    Acts as the main entrypoint to the application

    Runs the flask app where the webapp can be reached
"""
from app import app
import os.path

if __name__ == "__main__":
    if not os.path.exists('models/chatbot_model.keras'):
        raise RuntimeError(
            "Train the model before starting the application: model not found")
    app.run()
