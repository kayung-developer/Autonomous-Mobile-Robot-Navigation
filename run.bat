@echo off
:: run.bat - A script to set up and run the robot simulator on Windows

echo Checking for Python...
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.8+ and ensure it's added to your system PATH.
    pause
    exit /b
)

echo Creating virtual environment in ".\venv"...
python -m venv venv

echo Activating environment and installing dependencies...
call venv\Scripts\activate.bat

pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies. Please check your network and pip setup.
    pause
    exit /b
)

echo Starting the Robot Navigation Simulator...
python main.py

echo Application closed.
pause