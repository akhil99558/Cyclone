# Cyclone Intensity Prediction App

This repository contains the source code for a Streamlit web application that predicts cyclone intensity from infrared images.

## Overview

The Cyclone Intensity Prediction App analyzes infrared images of cyclones and estimates their intensity. It leverages a pre-trained convolutional neural network (CNN) model to perform the intensity prediction.

## Model Used

The deep learning model used in this application is based on a SimpleCNN architecture. It consists of convolutional layers followed by max-pooling layers and fully connected layers. The model is trained on a dataset of infrared images of cyclones to predict cyclone intensity based on the extracted features.

## Dataset

The dataset used to train the model is obtained from Kaggle and consists of Infrared Raw Cyclone Images captured by the INSAT-3D satellite from 2013 to 2021.

Dataset Source: [INSAT-3D Infrared Raw Cyclone Images (2013-2021)](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021)

## Usage

To run the Cyclone Intensity Prediction App locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/akhil99558/Cyclone.git
2.Navigate to the project directory and install dependencies:

  '''bash
      cd Cyclone
      pip install -r requirements.txt

3.Run the Streamlit app:
    '''bash
       python -m streamlit run app.py
## NOTE: Still some issues to fix

       
