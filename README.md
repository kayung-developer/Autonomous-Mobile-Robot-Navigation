# Advanced Python Autonomous Mobile Robot Navigation Program

This project demonstrates an advanced autonomous mobile robot navigation program using CustomTkinter for the GUI, incorporating machine learning (ML) and computer vision (CV) for a sophisticated navigation system.

## Table of Contents

- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Setup](#docker-setup)
- [Contributing](#contributing)
- [License](#license)

## Features

- Autonomous robot navigation using advanced path planning algorithms
- Obstacle detection and avoidance using ML and CV
- Real-time data capture from camera and LIDAR sensors
- GUI for monitoring and controlling the navigation

## Technologies

- Python 3.9
- CustomTkinter
- OpenCV
- TensorFlow
- Numpy
- Docker

## Installation

### Prerequisites

- Python 3.9+
- Docker

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/kayung-developer/Autonomous-Mobile-Robot-Navigation.git
   
   cd robot-navigation
2.  Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the dependencies:

```bash
pip install -r requirements.txt

```

4. Usage
- Running the GUI
- Start the CustomTkinter GUI:
```bash
python gui/navigation_gui.py
```
- Running the Navigation System, Start the robot navigation system
```bash
python robot/navigation.py

```

5. Docker Setup
- Build the Docker image:
```bash
docker build -t robot-navigation .
```

- Run the Docker container:

```bash
docker run -p 8000:8000 robot-navigation
```

<b><center> Developed By Pascal Saviour</center></b>
