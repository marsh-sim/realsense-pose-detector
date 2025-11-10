# RealSense Pose Detector

Detect human body pose for MARSH using RealSense camera and MediaPipe model

## Install & Run

The project is managed with [UV](https://docs.astral.sh/uv/getting-started/installation/#winget)

```sh
uv run pose.py
```

## Full-Body Pose Landmark Model (BlazePose Tracker)

The landmark model currently included in MediaPipe Pose predicts the location of 33 full-body landmarks (see figure below), each with (`x, y, z, visibility`).
The current model does not use depth data directly, we're projecting the positions identified on 2D image into 3D space using RealSense afterwards.
This means that the position of points is centered on the joint in 2D, but on the closest surface to camera in depth.

![Pose Description](readme/pose_tracking_full_body_landmarks.png)

*[Reference: Pose landmark detection guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)*

### About
MIT License - Copyright (c) 2021 Florian Bruggisser, Copyright (c) MARSH-Sim contributors

Based on the ideas of [mediapipe-osc](https://github.com/cansik/mediapipe-osc).
