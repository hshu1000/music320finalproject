# Pose-to-Wave: Real-Time Gesture-to-Sound Synthesis

This repository contains the full codebase for **Pose-to-Wave**, a real-time gesture-controlled audio synthesizer.

The system uses:

- **YOLOv8-Pose** and **MediaPipe Hands** for real-time pose tracking  
- A custom **additive harmonic synthesizer**  
- Real-time waveform + camera feed visualizations  
- Terminal controls for instruments, musical scale, pedal mode, and more.

---

## 📦 1. Installation & Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 1.2 Create the Conda Environment

You must create the environment using the provided environment.yml file

```bash
conda env create -f environment.yml
conda activate music320finalproject
```

If the environment.yml changes later:

```bash 
conda env update -f environment.yml --prune
```

## 2. Project Structure
```bash
.
├── environment.yml
├── pose_synth_launcher.py   # Main launcher with terminal user controls
├── pose_detect.py           # YOLO + MediaPipe arm & hand pose tracking
├── freq_processing.py       # Additive synthesis + pedal/echo DSP
├── plotter.py               # Waveform + camera + overlays plotting
├── diagrams.py              # System and concept diagrams
└── README.md
```

## 3. Running the System

Launch the main application:
```bash
python pose_synth_launcher.py
```

This will open:
- Webcam feed with pose overlays
- Audio synthesis engine
- Waveform visualizer
- Terminal command interface

## 4. Terminal Command Reference

When the program starts, you'll see:

```bash
=== Pose Synth Terminal Control ===
Commands:
  mode hand                 -> use MediaPipe hand tracking
  mode arm                  -> use YOLO arm pose tracking
  scale <name>              -> set musical scale (e.g. "c major")
  instrument personN <name> -> set instrument for person index N
  pedal on/off              -> enable or disable 10-second reverb tail
  pedal time <sec>          -> adjust reverb/echo duration
  quit                      -> exit program
```

## 5. System Requirements

- macOS or Linux recommended
- A webcam
- Low-latency audio output
- Python 3.9 (installed via environment.yml)

## 6. Contact

Luke Qiao
Stanford University
lkqiao@stanford.edu

Hannah Shu
Stanford University
hshu100@stanford.edu