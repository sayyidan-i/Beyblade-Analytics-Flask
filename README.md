# Beyblade Battle Analysis System

## Overview
The Beyblade Battle Analysis System is designed to analyze and interpret Beyblade battles using advanced computer vision techniques and machine learning models. This system captures video of battles, detects the Beyblades, and provides detailed insights through a user-friendly dashboard.

## Tech Stack
- **Flask**: A lightweight web framework used to build the dashboard and handle video uploads and user interactions.
- **OpenCV**: Used for image processing and video analysis.
- **Ultralytics YOLO**: A state-of-the-art object detection model utilized to detect and track Beyblades in the video.
- **Gemini**: An advanced generative AI model used to analyze battle logs and provide insights based on the results.
- **Pandas**: Used for data manipulation and to save battle logs and results to CSV files.
- **Roboflow**: A tool used for manual labeling of training data.

## Model Selection
The Beyblade detection model chosen for this project is the YOLOv10 (You Only Look Once version 10) model. This model was selected due to its real-time object detection capabilities, high accuracy, and efficiency in processing video frames. 

The training data for this model was sourced from YouTube videos of Beyblade battles, where key frames were extracted and manually labeled using Roboflow. This allowed for a comprehensive dataset that captures a variety of Beyblade types and battle scenarios.

## Model Performance
The performance of the Beyblade detection model is as follows:
- **Precision**: 97.88%
- **Recall**: 94.04%
- **mAP50**: 98.15%
- **mAP50-95**: 84.12%
- **Fitness**: 85.53%

These metrics indicate that the model performs exceptionally well in detecting and recognizing Beyblades in various battle conditions.

## Additional Data Generation Logic
In addition to detecting Beyblades, the system generates several key insights during the analysis of each battle:

1. **Collision Detection**: The system calculates the distance between the detected Beyblade bounding boxes to determine if a collision has occurred. This is crucial for understanding the dynamics of the battle.
  
2. **Stopping Condition**: The program utilizes optical flow techniques to monitor the movement of the Beyblades. If the average motion falls below a specified threshold for a set number of frames, the Beyblade is marked as "stopped."
  
3. **Out of Arena Detection**: The system tracks the number of frames a Beyblade goes undetected. If a Beyblade is missing for a certain duration, it is considered "out of the arena," and the opponent is declared the winner.

4. **Battle Analysis**: The battle logs, which contain frame-by-frame data on the positions and statuses of the Beyblades, are analyzed using Gemini. This analysis provides a detailed report on the battle, including insights on the performance of each Beyblade and the overall outcome.

## Conclusion
This system combines computer vision and machine learning to provide an in-depth analysis of Beyblade battles, making it an invaluable tool for fans and competitive players alike. With its user-friendly interface and comprehensive insights, the Beyblade Battle Analysis System enhances the viewing experience and offers actionable data for improving performance in future battles.