## ASD CTF 2024 - "Data Science" Flag
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
