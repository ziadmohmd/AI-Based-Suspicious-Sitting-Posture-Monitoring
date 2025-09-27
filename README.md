# Real-time Posture Monitoring

An AI-powered posture monitoring system designed to enhance workplace ergonomics and employee well-being. Utilizing advanced computer vision techniques, PostureGuard Pro provides real-time posture analysis, eye state detection, employee recognition, and comprehensive data logging with interactive dashboards.

## Features

- **Real-time Posture Monitoring**: Detects and classifies posture states including Good, Slouched, Leaning, Away, and Inactive using MediaPipe Pose estimation.
- **Eye State Detection**: Monitors eye openness using MediaPipe Face Mesh to identify inactivity or closed eyes.
- **Employee Recognition**: Integrates InsightFace for face recognition to identify and track individual employees.
- **Data Logging**: Automatically logs posture data, body angles, and timestamps to a CSV file for historical analysis.
- **Interactive Dashboards**: Generates visual charts and graphs using Plotly for posture frequency, eye states, body angles, and time-series data.
- **Notifications**: Sends desktop notifications for prolonged bad posture or inactivity using Plyer.
- **Admin Panel**: Web-based interface for employee registration, model training, and dashboard management.
- **Live Video Feed**: MJPEG streaming of the camera feed with overlaid posture information.
- **Theme Support**: Dark and light mode toggle for user interface customization.
- **Cross-platform**: Runs on Windows, macOS, and Linux with camera support.

## Tech Stack

- **Backend**: Python 3.x, Flask
- **Computer Vision**: OpenCV, MediaPipe (Pose and Face Mesh)
- **Face Recognition**: InsightFace with ArcFace model
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly
- **Database**: SQLite for employee data
- **Frontend**: HTML, CSS, JavaScript (Vanilla JS)
- **Notifications**: Plyer
- **Deployment**: Local Flask server

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Internet connection for initial model downloads

### Steps

1. **Clone or Download the Repository**:
   ```
   git clone https://github.com/your-repo/postureguard-pro.git
   cd postureguard-pro
   ```

2. **Create a Virtual Environment** (Recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   ```
   python mainapp.py
   ```



## Usage

### Main Interface

1. Click "Start" to begin monitoring.
2. The video feed will display with real-time posture overlays.
3. View live metrics including posture state, eye state, employee name, durations, and body angles.
4. Use the dashboard section to filter and view historical data charts.
5. Click "Capture Dashboard" to save a snapshot of the current dashboard.

### Admin Panel

1. **Register Employees**:
   - Enter employee name.
   - Capture face using the camera.
   - Click "Register" to add to the database.
   - Click "Train Model" to update the face recognition model.

2. **Manage Employees**:
   - View list of registered employees.
   - Rename or remove employees as needed.

3. **View Dashboards**:
   - Select an employee and click "View Dashboard" to open personalized charts.

### Notifications

- Desktop notifications will appear for:
  - Inactivity or closed eyes for more than 5 seconds (cooldown 60s).
  - Bad posture (Slouched/Leaning) for more than 20 seconds (cooldown 60s).
  - Employee away from workstation for extended periods.

## API Endpoints

- `GET /`: Main monitoring interface.
- `GET /admin`: Admin panel.
- `POST /register_employee`: Register a new employee (JSON: {name, image}).
- `POST /train_model`: Train the face recognition model.
- `GET /employees`: Retrieve list of employees.
- `POST /start`: Start monitoring.
- `POST /stop`: Stop monitoring.
- `GET /video_feed`: MJPEG video stream.
- `GET /metrics`: Current metrics (JSON).
- `GET /dashboard_html`: Dashboard HTML with charts.
- `POST /save_dashboard`: Save dashboard snapshot.
- `GET /debug_csv`: Debug CSV data.

## Configuration

Key configuration variables in `mainapp.py`:

- `LOG_INTERVAL_SEC`: Logging interval (default 3s).
- `STATE_NOTIFY_THRESHOLD`: Notification threshold for inactivity (5s).
- `POSTURE_NOTIFY_THRESHOLD`: Notification threshold for bad posture (20s).
- `EAR_THRESHOLD`: Eye aspect ratio threshold for closed eyes (0.28).
- `FRAME_WIDTH`: Output frame width (960px).
- `CSV_FILE`: Path to posture history CSV.
- `DB_FILE`: Path to employee database.

## Data Files

- `posture_history.csv`: Logged posture data.
- `employees.db`: SQLite database for employee embeddings.
- `captured_dashboards/`: Saved dashboard HTML snapshots.
- `static/face_data/`: Stored face images for employees.

## Troubleshooting

- **Camera Issues**: Ensure camera permissions are granted. Try different camera indices in the code.
- **Model Training**: Ensure face images are clear and well-lit for better recognition.
- **Notifications**: Plyer may require additional setup on some systems.
- **Performance**: Reduce frame rate or resolution if experiencing lag.



## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

This tool is for educational and ergonomic purposes. Ensure compliance with privacy laws when using face recognition features.



