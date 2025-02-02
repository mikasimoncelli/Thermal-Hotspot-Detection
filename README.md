# 🎯 Thermal Hotspot Detection

## Overview
This Python script processes thermal camera footage from railway tracks to detect and highlight temperature hotspots. It uses OpenCV to analyze video frames, identify areas of elevated temperature (shown as red/purple regions), and outputs an annotated video with detection boxes and status information.

## Features
- **Automated Detection** - Identifies thermal hotspots in grayscale footage
- **Real-time Visualization** - Draws boxes around detected hotspots
- **Status Tracking** - Shows frame numbers and detection status
- **Gauge Filtering** - Automatically ignores temperature gauge overlay
- **Intelligent Merging** - Combines nearby detections to avoid duplicates
- **Detection Summary** - Generates text file with hotspot sequence information
- **Full Video Output** - Maintains original video length with annotations

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or newer

### Installation
```bash
# Clone the repository
git clone https://github.com/mikasimoncelli/Thermal-Hotspot-Detection
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### Usage
```bash
# Basic usage (output will be "Thermal Clip 1_annotated.mp4")
python detect_hotspots.py "Thermal Clip 1.mp4"

# Specify custom output path
python detect_hotspots.py "Thermal Clip 1.mp4" -o "Custom Output.mp4"

# Show help
python detect_hotspots.py --help
```

## How It Works
The script processes thermal footage in the following steps:
- Converts frames to HSV color space for color detection
- Identifies red/purple regions indicating hotspots
- Filters out the temperature gauge overlay
- Merges nearby detections to avoid duplicates
- Outputs annotated video with frame numbers and status

## Input Requirements
- MP4 video file from thermal camera
- Grayscale footage with red/purple hotspots
- Temperature gauge overlay (automatically filtered)

## Output Format
### Video Output (input_video_annotated.mp4)
- Green boxes around detected hotspots
- Frame counter in bottom right
- Color-coded status (red for detected, green for clear)
- Original video length preserved

### Summary Output (input_video_summary.txt)
- List of detected hotspot sequences
- Frame ranges for each detection
- Number of frames in each sequence

---