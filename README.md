# 😴 Drowsiness Detection System

A real-time drowsiness and yawn detection system built with Python, OpenCV, and `face_recognition`. It monitors a driver's eyes and mouth via webcam (or mobile camera) and triggers voice alerts when signs of drowsiness or yawning are detected.

---

## Features

- **Real-time Eye Tracking** — Calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure.
- **Yawn Detection** — Measures lip distance to identify yawning.
- **Voice Alerts** — Uses `espeak` to audibly warn the user when drowsiness or yawning is detected.
- **Dual Camera Support** — Works with a local webcam or a remote mobile camera (via IP Webcam app).
- **Cross-Platform** — Includes macOS-specific camera backend handling alongside general support.
- **Configurable Thresholds** — Easily tune sensitivity for eye closure duration, yawn distance, frame rate, and more.

---

## How It Works

1. Captures video frames from a webcam or mobile IP camera.
2. Detects faces and facial landmarks using the `face_recognition` library.
3. Computes the **Eye Aspect Ratio (EAR)** — if it falls below a threshold for a sustained number of frames, a drowsiness alert is triggered.
4. Computes the **lip distance** — if it exceeds a threshold, a yawn alert is triggered.
5. Visual overlays (eye/lip contours, EAR/yawn metrics) are drawn on the live video feed.
6. Voice alarms are played via `espeak` in background threads.

---

## Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| Language            | Python 3                            |
| Computer Vision     | OpenCV, imutils                     |
| Face Detection      | face_recognition (dlib-based)       |
| Voice Alerts        | espeak (system TTS)                 |
| Threading           | Python `threading` module           |

---

## Prerequisites

- **Python 3.7+**
- **espeak** installed on your system (for voice alerts)
  - macOS: `brew install espeak`
  - Ubuntu/Debian: `sudo apt-get install espeak`
- A working **webcam**, or a mobile device running the **IP Webcam** app (for remote camera mode)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Install Python dependencies**

   ```bash
   pip install opencv-python numpy imutils face_recognition
   ```

   > **Note:** `face_recognition` requires `dlib`, which may need CMake and a C++ compiler. See the [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation) for details.

---

## Usage

### Using your webcam (default)

```bash
python i.py
```

### Using a mobile camera (IP Webcam)

1. Install the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app on your phone.
2. Start the server in the app and note the IP address.
3. Update `MOBILE_IP` and `MOBILE_PORT` in the `CONFIG` dictionary inside `i.py` if they differ from the defaults.
4. Run:

   ```bash
   python i.py --iphone
   ```

### Controls

| Key       | Action          |
|-----------|-----------------|
| `k` / `ESC` | Quit the application |
| `q`       | Quit the application |

---

## Configuration

All tunable parameters are in the `CONFIG` dictionary at the top of `i.py`:

| Parameter               | Default   | Description                                         |
|--------------------------|-----------|-----------------------------------------------------|
| `EYE_AR_THRESH`          | `0.3`     | EAR threshold below which eyes are considered closed |
| `EYE_AR_CONSEC_FRAMES`   | `30`      | Consecutive frames below threshold to trigger alarm  |
| `YAWN_THRESH`            | `20`      | Lip distance threshold to detect a yawn              |
| `FRAME_WIDTH`            | `450`     | Width to resize each frame for processing            |
| `FPS`                    | `30`      | Target frames per second                             |
| `MOBILE_IP`              | `172.20.10.4` | IP address of the mobile camera                  |
| `MOBILE_PORT`            | `8080`    | Port for the mobile camera stream                    |

---

## Project Structure

```
Drowsiness Detection/
├── i.py          # Main application — detection logic, camera handling, alerts
└── README.md     # Project documentation
```

---

## License

This project is for academic/educational purposes.
