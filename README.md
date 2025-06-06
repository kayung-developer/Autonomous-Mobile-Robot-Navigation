# Intellibot Mobile Robot Navigation

This project is a comprehensive, self-contained simulator for an autonomous mobile robot. It features a FastAPI backend for the simulation logic and a CustomTkinter GUI for real-time visualization and interactive control. The entire application runs from a single Python script, with the backend and frontend communicating seamlessly via a REST API and WebSockets.

![screenshot2](https://github.com/user-attachments/assets/b52c513c-72ea-45da-a115-cca2f1ad7293)
![Screenshoot](https://github.com/user-attachments/assets/7d527847-683b-475d-aa29-654d16bbe611)



## Key Features

- **Robot Simulation**:
    - Kinematics model with acceleration and speed limits.
    - Odometry simulation with configurable, cumulative noise.
    - Fused pose estimation (simulated).
    - Battery model with drain based on activity (movement, sensors).

- **Environment & Path Planning**:
    - Configurable 2D grid map with static obstacles.
    - Dynamic obstacles that follow predefined paths.
    - Variable terrain costs affecting pathfinding.
    - **A* Path Planning** with path smoothing and a line-of-sight shortcut algorithm for more efficient routes.

- **Sensor Simulation**:
    - **LIDAR**: Simulates a 2D laser scanner with configurable FOV, range, and noise.
    - **Camera**: Simulates a forward-facing camera that "detects" dynamic obstacles within its view, complete with bounding boxes and confidence scores.

- **Autonomous Navigation & Control**:
    - Proportional controller for path following.
    - Dynamic path replanning if the current path is obstructed.
    - LIDAR-based emergency stop to prevent collisions with unforeseen obstacles.
    - Seamless switching between autonomous and manual control modes.

- **Interactive GUI (CustomTkinter)**:
    - Real-time visualization of the map, obstacles, and robot.
    - Displays multiple robot poses: ground truth, noisy odometry, and fused pose.
    - Visualizes the planned path, LIDAR scan, camera detections, and camera FOV.
    - **Map Editor**: Add/remove obstacles or set terrain costs with mouse clicks.
    - **Full Robot Control**: Set navigation goals, start/stop autonomous mode, and take direct manual control with sliders or a floating joypad.

- **Backend Architecture (FastAPI)**:
    - Runs in a background thread, handling all simulation logic.
    - Provides REST endpoints for commands (e.g., set goal, manual control).
    - Uses WebSockets to stream real-time state data to the GUI, ensuring a responsive and low-latency display.

## Getting Started

Follow these steps to get the simulator running on your local machine.

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.8 or newer.

### Installation & Running

#### Option 1: Using the Convenience Scripts (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd robot-navigation-simulator
    ```

2.  **Run the script for your OS:**
    -   **On Linux/macOS:**
        ```bash
        # Make the script executable first
        chmod +x run.sh
        ./run.sh
        ```
    -   **On Windows:**
        ```bash
        run.bat
        ```
    These scripts will automatically create a virtual environment, install the dependencies, and launch the application.

#### Option 2: Manual Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kayung-developer/Intellibot-Mobile-Robot-Navigation.git
    cd robot-navigation
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python robotics.py
    ```
    The FastAPI server will start in the background, and the GUI window will appear.

## How to Use the Simulator

- **Main Map View**:
    - **Dark Gray Rectangles**: Static obstacles.
    - **Purple Rectangles**: Dynamic obstacles.
    - **Blue Path**: The robot's planned route. The cyan circle is the next waypoint.
    - **Gold Triangle**: The final goal position.
    - **Robot Poses**:
        - `Fused Pose (Blue)`: The robot's main, "best estimate" position. The arrow shows its orientation.
        - `Odometry Pose (Orange)`: The position calculated from wheel encoders, which drifts over time.
        - `True Pose (Red, Transparent)`: The actual, perfect position of the robot in the simulation.
    - **Camera FOV**: The transparent yellow cone in front of the robot.

- **Setting a Goal & Navigating**:
    1.  In the **Map Editor** panel on the right, ensure **"Set Goal"** is selected.
    2.  **Left-click** on a valid (unobstructed) location on the map. A gold triangle will appear.
    3.  Click the **"Start Autonomous"** button in the **Robot Control** panel. The robot will plan a path and start moving.
    4.  Click **"Stop Autonomous"** to halt navigation.

- **Manual Control**:
    - **Floating Joypad**: Use the `▲ ▼ ◄ ►` buttons on the map for quick directional control. The center `●` button is a hard stop.
    - **Sliders**: For more precise control, use the **Linear** and **Angular** speed sliders in the **Robot Control** panel and click **"Send Manual Speeds"**.
    - Taking manual control will automatically cancel autonomous navigation.

- **Map Editing**:
    - **Add Obstacle**: Select this mode and left-click on the map to add a 1x1 static wall.
    - **Remove Obstacle**: Select this mode and left-click on a wall or terrain to remove it.
    - **Set Terrain Cost**: **Right-click** anywhere on the map to open a dialog and set a movement cost for that cell (higher cost makes the path planner avoid it).

## Code Structure

The `main.py` file is organized into several key sections and classes:

-   **Configuration (`CONFIG`)**: A global dictionary holding all simulation parameters.
-   **Backend (Simulation & API)**:
    -   `Robot`: Models the robot's state, physics, and battery.
    -   `Environment`: Manages the map grid, obstacles, and terrain costs.
    -   `PathPlanner`: Implements the A* algorithm for finding paths.
    -   `robot_simulation_loop()`: The core async function that runs the entire simulation tick by tick.
    -   **FastAPI Endpoints**: Functions decorated with `@app.*` that handle HTTP requests from the GUI.
-   **Frontend (GUI)**:
    -   `RobotNavigationApp`: The main CustomTkinter application class.
    -   `_ws_client_thread_func`: Manages the WebSocket client connection in a background thread.
    -   `update_gui_from_state`: Parses incoming WebSocket data and updates all GUI elements.
    -   `redraw_*` methods: Handle drawing on the Tkinter canvases.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
