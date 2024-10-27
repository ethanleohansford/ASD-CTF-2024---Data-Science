import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pickle
import re
import time

# Define image size
img_height, img_width = 224, 224

# Load the trained model
model = load_model('clock_time_classifier.h5')

# Load the class indices
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
class_indices = {v: k for k, v in class_indices.items()}

# Function to predict the time from a new image
def predict_time(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    confidence_score = np.max(prediction) * 100  # Confidence score as a percentage
    predicted_time = class_indices[predicted_class[0]]

    # Ensure the predicted time is in the correct format (HH:MM)
    match = re.match(r'(\d{1,2})[:\\-](\d{2})', predicted_time)
    if match:
        hours, minutes = match.groups()
        predicted_time = f"{int(hours):02}:{int(minutes):02}"

    return predicted_time, confidence_score

# Natural sorting function to handle numerical parts in filenames
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

# Function to predict times for all images in a directory and return the sorted order
def predict_times_in_folder(folder_path):
    results = []
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")], key=natural_sort_key)
    times = []

    for index, filename in enumerate(filenames):
        image_path = os.path.join(folder_path, filename)
        predicted_time, confidence_score = predict_time(image_path)
        results.append({
            "Index": index,
            "Clock Image": filename,
            "Predicted Time": predicted_time,
            "Confidence Score (%)": confidence_score
        })
        times.append(predicted_time)

    # Create a DataFrame
    results_df = pd.DataFrame(results)
    
    # Convert times to minutes since midnight for correct sorting
    def time_to_minutes(time_str):
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    results_df['Time in Minutes'] = results_df['Predicted Time'].apply(time_to_minutes)
    
    # Sort by 'Time in Minutes'
    sorted_df = results_df.sort_values(by='Time in Minutes').reset_index(drop=True)
    
    # Extract sorted indices
    sorted_indices = sorted_df['Index'].tolist()
    sorted_order = ','.join(map(str, sorted_indices))

    # Print the DataFrame
    print(results_df)
    print(sorted_df)

    return sorted_order

# Example usage
if __name__ == "__main__":
    start_time = time.time()  # Start timing
    folder_path = 'images'
    sorted_order = predict_times_in_folder(folder_path)
    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Sorted order: {sorted_order}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
