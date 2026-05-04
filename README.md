<h1 align="center"> drone-aruco-localization </h1>

<p align="center">
  <img src="docs/camera-drone.png" width="150"/>
</p>

<p align="center"> Estimation of a current drone position based on ArUco markers </p>

## Project description
The position estimation process is based on establishing correspondences between points in the real environment (represented by ArUco markers with known 3D coordinates and their 2D projections in the image plane.
For each frame froma video sequence recorded by a drone camera: (i) detect and identify the ArUco markers,(ii) estimate the drone position (i.e. the drone-mounted camera position) using the camera matrix.

## Project structure
- config/ → parameters (e.g. camera calibration)
- data/ → input and output data
  - proccesed/ → proccesed files
  - raw/ → raw CSV and MP4 files
- docs/ → all documentation (PDFs, reports, diagrams)
- src/ → source code (e.g. OpenCV, ArUco, position estimation)
- Dockerfile → in the root directory (standard)

## Prepare input data
Copy your recorded MP4 files into the following directory:

```bash
data/raw/
```
Example
```bash
cp /path/to/your/videos/*.MP4 data/raw/
```

## Build and Run Docker container

To build the Docker image based on the provided Dockerfile, run:
```bash
docker build -t drone-aruco-localization .
```
### Run container (macOS with XQuartz)

Before starting the container, allow connections to the X server (run this in your host terminal, not inside Docker):
```bash
xhost +
```
Then start the container:
```bash
docker run -it --rm \
  -e DISPLAY=docker.for.mac.host.internal:0 \
  drone-aruco-localization
```