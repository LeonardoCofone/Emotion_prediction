# Emotion Recognition AI  
**Real-time emotion classification from face images using deep learning**

---

## 1 · Introduction  

This repository contains the full source code and documentation for a real-time emotion recognition system built entirely by **Leonardo Cofone**.

The project includes:  
- Face detection using MediaPipe  
- Emotion classification using a custom-trained TensorFlow CNN model  
- Real-time webcam inference with a user-friendly graphical interface built with Kivy  
- Image upload and prediction functionality with optimized class-specific thresholds  

The goal is to provide an easy-to-use, efficient application for emotion detection on typical desktop setups, offering clear visual feedback and reliable predictions.

---

## 2 · Dataset - FER2013  

| Feature       | Value                                         |
| ------------- | --------------------------------------------- |
| Origin        | The model was trained on a large curated dataset of facial images labeled with six emotions (Angry, Fear, Happy, Sad, Surprise, Neutral). Due to size and licensing restrictions, the dataset is not publicly included. |
| Size          | About 25000 images |
| Classes       | 6 classes: Angry, Fear, Happy, Sad, Surprise, Neutral |

### Notes  
- The trained model (`Emotion_detection_final2.h5`) is **not included** here due to its large size.  
- Optimized probability thresholds per emotion class are implemented to improve prediction accuracy and F1 scores.

---

## 3 · Project Structure  
├── GUI.py                                  # Main application script with GUI and live camera  
├── model/  
│   ├── Emotion_detection_final2.h5         # Pre-trained TensorFlow model (not included)  
│   └── best_threshold.txt                  # Classification threshold  
├── assets/  
│   └── LOGO.png                            # Logo image used in loading screen, you can change it ;)   
├── README.md                               # This documentation file  
├── LICENSE.txt                             # The licensa (MIT)  
├── Creation_of_the_model.ipynb             # The notebook used to create the model

---

## 4 · How It Works  

- **Face Detection**: MediaPipe detects faces from input images or webcam frames.  
- **Preprocessing**: Faces are cropped, converted to grayscale, resized to 160x160 pixels, and normalized.  
- **Emotion Prediction**: The CNN model outputs probabilities for each emotion class.  
- **Thresholding**: Class-specific probability thresholds filter out low-confidence predictions to improve precision and recall.  
- **Visualization**: Bounding boxes and predicted labels are drawn on the GUI for user feedback.

---

## Thank you for watching!
> If you have something to report, contact me
---
LINK: [Kggle](https://www.kaggle.com/zlatan599)  
LINK: [Linkedin](https://www.linkedin.com/in/leonardo-cofone-914228361/)
LINK:
[LinkTree](https://linktr.ee/Leonardo_Cofone)
