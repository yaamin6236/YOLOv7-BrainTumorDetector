# YOLOv7-BrainTumorDetector

## Brain Tumor Detector

A computer vision application designed to detect brain tumors in MRI scans using the YOLOv7 model.

## Overview

This application uses a dataset from Roboflow, which has been augmented and trained on Google Colab using the YOLOv7 model on a Tesla V100 GPU. The application can process both images and videos to detect and highlight potential brain tumors.

## Features

- **Image Processing**: Detect brain tumors in MRI images.
- **Video Processing**: Process video footage to detect tumors in real-time MRI scans.
- **Visualization**: Highlights detected tumors with bounding boxes and labels.

## Technologies Used

- **OpenCV**: For image and video processing.
- **YOLOv7**: For object detection.
- **Roboflow**: For the dataset.
- **Google Colab**: For training the model on a Tesla V100 GPU.

## Setup and Usage

1. Ensure you have OpenCV installed in your environment.
2. Clone this repository.
3. Place your MRI images or videos in the desired directory.
4. Update the paths in the code for model weights, model configuration, and input files.
5. Run the program and choose between 'image' or 'video' processing when prompted.
6. View the processed results with detected tumors highlighted.

## Code Structure

- **Main Function**: Allows users to choose between processing an image or a video.
- **Video Function**: Processes video footage to detect tumors.
- **Image Function**: Processes individual MRI images to detect tumors.
- **Utility Functions**: Helper functions for image conversions and other utilities.

## Future Enhancements

- Improve detection accuracy with more training data.
- Implement a GUI for easier user interaction.
- Extend the application to detect other abnormalities in MRI scans.
