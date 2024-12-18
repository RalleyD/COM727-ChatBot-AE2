#!/usr/bin/env/python3

"""
    main.py

    Acts as the main entrypoint to the application

    Runs the flask app where the webapp can be reached
"""
from app import app

if __name__ == "__main__":
    app.run()
