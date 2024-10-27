import os
import socket
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import pandas as pd
import pickle
import re
import time

# Define image size and model paths
img_height, img_width = 224, 224
model_path = 'clock_time_classifier.h5'
class_indices_path = 'class_indices.pkl'

# Load the trained model
model = load_model(model_path)

# Load the class indices
with open(class_indices_path, 'rb') as f:
    class_indices = pickle.load(f)
class_indices = {v: k for k, v in class_indices.items()}

# Function to predict the time from a new image
def predict_time(image):
    if image is None:
        return None, None
    img = image.resize((img_height, img_width))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

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

# Function to decode a base64 image and verify it
def decode_and_verify_image(b64_image):
    try:
        image_data = base64.b64decode(b64_image)
        image = Image.open(BytesIO(image_data))
        image.verify()  # Verify that it is an image
        image = Image.open(BytesIO(image_data))  # Reopen to reset the file pointer after verify
        return image
    except (base64.binascii.Error, IOError):
        print(f"Skipping invalid base64 image: {b64_image[:30]}...")
        return None

# Function to process chunks of data
def process_chunk(chunk, partial_data, output_dir, valid_image_count, results):
    data = partial_data + chunk
    base64_images = data.split("\n")
    
    # Keep the last part of the data as partial data if it might be incomplete
    partial_data = base64_images.pop() if data[-1] != "\n" else ""

    for b64_image in base64_images:
        if valid_image_count >= 12:
            break
        if b64_image.strip():
            image = decode_and_verify_image(b64_image)
            if image:
                # Save the image to the output directory
                valid_image_count += 1
                image_path = os.path.join(output_dir, f"image_{valid_image_count}.jpg")
                image.save(image_path)
                print(f"Saved image {valid_image_count} to {image_path}")

                # Predict the time
                predicted_time, confidence_score = predict_time(image)
                if predicted_time is not None:
                    # Append result to the results list
                    results.append({
                        "Index": valid_image_count - 1,
                        "Clock Image": f"image_{valid_image_count}.jpg",
                        "Predicted Time": predicted_time,
                        "Confidence Score (%)": confidence_score
                    })

    return partial_data, valid_image_count

def main():
    server_address = "54.79.244.247"
    port = 9875
    output_dir = "images"
    results = []

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            # Measure time taken to connect to the server
            connect_start_time = time.time()
            sock.connect((server_address, port))
            connect_end_time = time.time()
            print(f"Connected to {server_address} on port {port}")
            connect_time = connect_end_time - connect_start_time
            print(f"Time taken to connect: {connect_time:.2f} seconds")

            valid_image_count = 0
            partial_data = ""
            received_data = b""

            # Measure time taken to receive and process data from the server
            receive_start_time = time.time()
            while valid_image_count < 12:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                received_data += chunk
                partial_data, valid_image_count = process_chunk(chunk.decode('utf-8'), partial_data, output_dir, valid_image_count, results)
            receive_end_time = time.time()
            receive_time = receive_end_time - receive_start_time
            print(f"Time taken to receive and process data: {receive_time:.2f} seconds")

            # Create a DataFrame from the results
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

            # Print the DataFrame before and after sorting
            print("Results DataFrame before sorting:")
            print(results_df)
            print("Results DataFrame after sorting:")
            print(sorted_df)

            # Print the sorted order
            print(f"Sorted order: {sorted_order}")

            # Check the time taken so far
            elapsed_time = time.time() - receive_start_time
            print(f"Time taken before sending the string: {elapsed_time:.2f} seconds")

            # Send the sorted order back to the server as if it was user input
            sock.sendall((sorted_order + '\n').encode('utf-8'))
            print("Sorted order sent successfully.")
            
            # Receive the server's response
            response = sock.recv(4096)
            response_text = response.decode('utf-8')
            print("Response from server:")
            print(response_text)  # Print the response from the server

            # Save the server's response to a new file
            response_file = "server_response.txt"
            with open(response_file, "w") as file:
                file.write(response_text)
            print(f"Server response saved to {response_file}")

        except Exception as e:
            print(f"Exception occurred: {e}")
        finally:
            elapsed_time = time.time() - receive_start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
