#!/bin/bash
# run.sh - A script to set up and run the robot simulator on Linux/macOS

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 is not installed or not in your PATH."
    exit 1
fi

echo "Creating a virtual environment in './venv'..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Please check your network connection and pip setup."
    exit 1
fi

echo "Starting the Robot Navigation Simulator..."
python3 main.py

# Deactivate the environment when the script is done
deactivate
echo "Application closed."