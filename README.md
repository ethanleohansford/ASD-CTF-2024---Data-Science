# ASD CTF 2024 - "Data Science" Flag
This repository contains the solution for the "Data Science" challenge from ASD CTF 2024. The challenge involves building a machine learning model to recognize and sort times displayed on analog clock images. The solution automates the process to meet a strict 3-second time constraint, making manual solutions infeasible.

## Challenge Overview
- **Challenge Name:** "Data Science"
- **Goal:** Predict times on analog clock images, sort them in ascending order, and send the sorted list back to a remote server to retrieve the flag.
- **Constraints:**
  - **Dynamic Data:** Each server connection sends a new set of images in random order.
  - **Strict Time Limit:** The solution must process the images and return results within 3 seconds, making real-time machine learning essential.

## Solution Approach
To solve this challenge, we developed a machine learning model that can:

1. Recognize the time displayed on analog clock images.
2. Sort these times in ascending order.
3. Return the sorted list of times back to the server within the given time limit.

My solution consists of two main components:

1. train_model.py - Script to train the machine learning model on provided clock image data.
2. main_code.py - Script to connect to the server, receive images, predict times, and return the sorted order of times.

## Features
- Clock Image Classifier: A Convolutional Neural Network (CNN) model trained to recognize times from analog clock images.
- Server Communication: main_code.py connects to the server to receive images, predicts times, and returns the sorted result.
- Prediction Logging: Logs the server's response, which contains the CTF flag, to server_response.txt.

## Getting Started
### Prerequisites
Make sure you have Python installed, and then install the required dependencies:

```bash
pip install -r requirements.txt
```
## Training the Model
To generate a model that can recognize times on clock images, follow these steps:

1. Place the provided dataset in a folder named `dataset/,` with subfolders organized by time labels (e.g., "03_15" for 3:15).

2. Run the following command to train the model:

  ```bash
  python train_model.py
  ```
  This will output two files:

  - `clock_time_classifier.h5`: The trained model.
  - `class_indices.pkl`: A dictionary mapping model outputs to time labels.

## Running the Client Script
Once the model is trained, you can use `main_code.py` to interact with the server:

1. Ensure that `clock_time_classifier.h5` and `class_indices.pkl` are in the same directory as `main_code.py`.

2. Run the client script:

```bash
python main_code.py
```
This script will:

- Connect to the server using the provided IP and port.
- Receive and decode a batch of base64-encoded images.
- Use the trained model to predict times from each clock image.
- Sort the predicted times in ascending order.
- Send the sorted list of indices back to the server and receive the flag.
3. Check `server_response.txt` for the server's response, which should contain the CTF flag.

## Example Output
After running `main_code.py`, you should see output like:

```vbnet
Connected to the server at IP: [server IP] and Port: [port]
Receiving images...
Predicting times...
Sorted order sent back to the server.
Flag received: ASDCTF{your_flag_here}
```
The server response is saved in server_response.txt for reference.
