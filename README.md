# Emochannel X

## Introduction
Emochannel X is a Streamlit application designed to detect emotions in text. 
It leverages a pre-trained machine learning model to classify text into various emotional categories and provides a user-friendly interface for interaction.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Installation
To install and run Emochannel X locally, follow these steps:

1. Clone the repository:
    ```sh
    git clone <https://github.com/kiuwiya/Emochannel.git>
    cd <Emochannel>
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

## Usage
1. Navigate to the Streamlit app in your web browser.
2. Input text into the provided text area.
3. Click the "Submit" button to get the emotion prediction.
4. View the detected emotion and its corresponding emoji.

## Features
- **Emotion Detection**: Detects emotions such as anger, disgust, fear, happy, joy, neutral, sad, sadness, shame, and surprise.
- **Interactive Interface**: User-friendly interface to input text and view predictions.
- **Visualization**: Uses various visualization libraries for displaying results.

## Dependencies
- streamlit
- pandas
- numpy
- altair
- WordCloud
- matplotlib
- plotly
- joblib

## Configuration
Ensure that the model file `text_emotion.pkl` is located in the `model` directory.

## Documentation
For detailed documentation, refer to the inline comments and docstrings within the `app.py` file.

## Examples
1. Run the application and navigate to the home page.
2. Enter a sample text such as "I am feeling great today!".
3. Observe the emotion detection result and the corresponding emoji.

## Troubleshooting
- **Model File Not Found**: Ensure the `text_emotion.pkl` file is in the correct directory.
- **Dependencies**: Verify that all dependencies are installed correctly using `pip list`.

## Contributors
- [Nuraina Sofea Binti Ibrahim](https://github.com/kiuwiya)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
