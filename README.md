# Facial_Emotion_Dection
Overview
This project is a Python-based Facial Emotion Detection system that uses a Convolutional Neural Network (CNN) model to detect and classify human emotions in real-time. The system processes a webcam feed, identifies faces, and predicts the emotion displayed by each face.

Features
Real-time emotion detection from webcam feed.
Seven emotion classes: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.
Uses a pre-trained CNN model for emotion classification.
Visualization of bounding boxes around detected faces along with their respective emotions.
Technologies Used
Python
TensorFlow/Keras
OpenCV
NumPy
Matplotlib
How It Works
Preprocessing:
Input video frames are converted to grayscale to simplify processing.
Faces are detected using Haar Cascades, and each face is cropped.
Emotion Classification:
The cropped face is resized to 48x48 pixels and passed through a trained CNN model.
The model predicts the probabilities for each emotion class, and the emotion with the highest probability is selected.
Visualization:
Bounding boxes and predicted emotions are displayed on the webcam feed.
Setup Instructions
Clone this repository:
bash
Copy code
git clone https://github.com/your-username/Facial_Emotion_Detection.git
cd Facial_Emotion_Detection
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Add the haarcascade_frontalface_default.xml file to the project directory if not included (download it from OpenCV GitHub).
Train the model or use pre-trained weights:
If training:
bash
Copy code
python emotions.py --mode train
For real-time emotion detection:
bash
Copy code
python emotions.py --mode display
Folder Structure
bash
Copy code
Facial_Emotion_Detection/
├── data/                        # Training and validation datasets
│   ├── train/                   # Training data
│   └── test/                    # Validation data
├── emotions.py                  # Main Python script
├── haarcascade_frontalface_default.xml  # Haar Cascade for face detection
├── model.h5                     # Pre-trained model weights
└── requirements.txt             # Dependencies file
Requirements
Python 3.x
TensorFlow/Keras
OpenCV
NumPy
Matplotlib
Emotion Classes
The system can classify the following emotions:

Angry 😠
Disgusted 🤢
Fearful 😨
Happy 😊
Neutral 😐
Sad 😢
Surprised 😲
Output
Real-time webcam feed with bounding boxes around faces and their respective detected emotions displayed as text.
Future Enhancements
Integration of more robust emotion datasets for improved accuracy.
Expansion to recognize subtle micro-expressions.
Integration with sentiment analysis for textual or audio inputs.
