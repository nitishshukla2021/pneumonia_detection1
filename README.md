🫁 Pneumonia Detection Using DenseNet121
📌 Project Overview

This project implements a deep learning–based pneumonia detection system using chest X-ray images.
A DenseNet121 transfer learning model is trained to classify X-ray images into Normal and Pneumonia categories.

The trained model is deployed as an interactive Streamlit web application, allowing users to upload chest X-ray images and receive real-time predictions.

🎯 Objectives

Detect pneumonia accurately from chest X-ray images

Apply transfer learning using DenseNet121

Improve model generalization using data augmentation

Deploy the trained model as a user-friendly web application

📂 Dataset

Source: Kaggle (Chest X-ray Images Dataset)

Classes:

Normal

Pneumonia

Format: JPEG / PNG images

🧠 Model Architecture

Base Model: DenseNet121 (pre-trained on ImageNet)

Approach: Transfer Learning

Custom Layers:

Global Average Pooling

Fully Connected Dense Layers

Sigmoid Activation for Binary Classification

DenseNet121 was chosen due to its efficient feature reuse and strong performance in medical image analysis.

🔄 Data Preprocessing & Augmentation

Data preprocessing and augmentation were performed using Keras ImageDataGenerator to enhance model robustness.

Preprocessing Steps:

Image resizing

Pixel value normalization

Batch generation

Data Augmentation Techniques:

Rotation

Width and height shifting

Zooming

Horizontal flipping

These techniques helped reduce overfitting and improve model generalization on unseen data.

⚙️ Tech Stack

Programming Language: Python

Deep Learning: TensorFlow, Keras

Model: DenseNet121

Data Handling: NumPy, Pandas

Visualization: Matplotlib

Web Framework: Streamlit

Version Control: Git & Git LFS

🚀 Features

Binary classification (Normal vs Pneumonia)

Transfer learning with DenseNet121

Data augmentation using ImageDataGenerator

Streamlit-based web interface

Real-time chest X-ray image prediction

Large model handling using Git LFS

🖥️ Streamlit Web Application

The Streamlit app provides a simple and intuitive interface where users can:

Upload a chest X-ray image

The image is preprocessed automatically

The trained DenseNet121 model performs inference

The prediction result is displayed instantly
📊 Model Performance

Achieved strong classification accuracy on validation data

Data augmentation significantly improved generalization

Model demonstrated reliable performance on unseen chest X-ray images

⚠️ Disclaimer

This project is developed for educational and research purposes only.
It is not intended for real-world clinical diagnosis or medical decision-making.

👤 Author

Nitish Shukla
BSc in Data Science

🔗 GitHub: https://github.com/nitishshukla2021

⭐ Acknowledgements

Kaggle for providing the chest X-ray dataset

TensorFlow and Keras documentation

Streamlit for easy web app deployment
