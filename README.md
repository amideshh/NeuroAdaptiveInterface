# NeuroAdaptiveInterface

## Neuro-Adaptive Interface for Cognitive Overload & Sensory Processing

A Python-based project that uses webcam-based facial landmarks to explore gaze direction, blinking, eye closure, and session-level visual signals. It includes a Streamlit dashboard for reviewing and comparing recorded sessions.

## Features

- Webcam-based facial landmark tracking
- Gaze direction and look-away signal tracking
- Blink and eye-closure event monitoring
- Session logging and signal analysis
- Streamlit dashboard with session overview and history
- Session comparison and exploratory signal review
- Data validation utilities

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Streamlit
- Matplotlib

## Project Files

- `face_demo.py` — webcam detection and signal tracking
- `dashboard.py` — interactive Streamlit dashboard
- `overload_analysis.py` — session signal analysis
- `validate_sessions.py` — session data validation, if included
- `validate_lookaway.py` — look-away validation, if included
- `requirements.txt` — Python dependencies
- `screenshots/` — project screenshots

## Installation

1. Install Python compatible with the project's dependencies.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Create and activate a virtual environment.
5. Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Camera Application

```bash
python face_demo.py
```

Allow webcam access when prompted.

## Run the Dashboard

```bash
python -m streamlit run dashboard.py
```

Open the local URL displayed in the terminal.

## Important Limitations

This project explores camera-derived signals such as gaze direction, blinking, look-away duration, and eye closure. These signals are not a clinically validated measure of cognitive overload and must not be treated as a medical diagnosis.

## Privacy

Review session logs and screenshots before publishing. Do not commit personal recordings, sensitive session data, passwords, API keys, or other private information.

## Author

Amidesh
