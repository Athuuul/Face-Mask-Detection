# Facial Recognition with Data Retrieval

## Project Overview

This project focuses on developing a **Face Mask Detection and Recognition System** that overcomes the challenges posed by the widespread use of face masks. With face masks becoming a common feature in daily life, traditional methods of identification face significant hurdles. Our system accurately detects face masks and retrieves associated data for identified individuals, such as name, address, blood type, and other pertinent information stored in a database.

## Key Features

- **Face Mask Detection**: Utilizes advanced computer vision techniques to detect face masks with high accuracy.
- **Facial Recognition**: Recognizes individuals even when they are wearing face masks.
- **Data Retrieval**: Retrieves relevant details (name, address, etc.) from a database for recognized individuals.
- **Mobile App Integration**: A mobile application serves as the centralized platform for users to interact with face mask detection, facial recognition, and data retrieval functionalities.

## Technologies Used

- **Programming Language**: Python
- **Computer Vision**: OpenCV
- **Deep Learning Framework**: TensorFlow
- **Database Management**: SQLite
- **Mobile App Development**: Flutter or React Native (Optional)
- **Image Processing**: Edge detection, histogram equalization
- **Model**: Convolutional Neural Networks (CNNs) with transfer learning, Mask R-CNN model from TensorFlow Model Zoo

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Athuuul/Face-Mask-Detection.git
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    python main.py
    ```

## Project Requisites

- Python 3.8
- OpenCV 4.5
- TensorFlow 2.5
- SQLite

## Usage

1. **Face Mask Detection**: The system detects if an individual is wearing a face mask.
2. **Facial Recognition**: Recognizes the individual even with a mask.
3. **Data Retrieval**: Retrieves associated data from the database.

## Novelty and Impact

- **Social Impact**: Enhances public health and safety by encouraging mask-wearing and supporting compliance with infection control protocols.
- **Security**: Aids in crime prevention and enhances surveillance effectiveness.
- **Accessibility**: The mobile app ensures ease of use across diverse settings.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. [COVID-19 Face Mask Detector with OpenCV, Keras, TensorFlow, and Deep Learning](https://pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
2. [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
3. [Image Inpainting with OpenCV and Python](https://pyimagesearch.com/2020/05/18/image-inpainting-with-opencv-and-python/)
