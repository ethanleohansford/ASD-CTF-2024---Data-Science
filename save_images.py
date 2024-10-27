import socket
import base64
import os
from PIL import Image
from io import BytesIO
import time

def decode_image(b64_image):
    if b64_image.strip():
        try:
            # Decode base64 string
            image_data = base64.b64decode(b64_image)
            # Verify the image data
            image = Image.open(BytesIO(image_data))
            image.verify()  # Verify that it is, in fact, an image
            return image_data
        except (base64.binascii.Error, IOError):
            print(f"Skipping invalid base64 image: {b64_image[:30]}...")
            return None
    return None

def process_chunk(chunk, partial_data, output_dir, valid_image_count):
    data = partial_data + chunk
    base64_images = data.split("\n")
    
    # Keep the last part of the data as partial data if it might be incomplete
    partial_data = base64_images.pop() if data[-1] != "\n" else ""

    for b64_image in base64_images:
        image_data = decode_image(b64_image)
        if image_data:
            valid_image_count += 1
            image_path = os.path.join(output_dir, f"image_{valid_image_count}.jpg")
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)
            print(f"Saved image {valid_image_count} to {image_path}")
        if valid_image_count >= 12:
            break

    return partial_data, valid_image_count

def main():
    server_address = "54.79.244.247"
    port = 9875
    output_dir = "images"

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Measure time taken to connect to the server
        connect_start_time = time.time()
        sock.connect((server_address, port))
        connect_end_time = time.time()
        print(f"Connected to {server_address} on port {port}")
        connect_time = connect_end_time - connect_start_time
        print(f"Time taken to connect: {connect_time:.2f} seconds")

        valid_image_count = 0
        partial_data = ""

        # Measure time taken to receive and process data from the server
        receive_start_time = time.time()
        while valid_image_count < 12:
            chunk = sock.recv(4096)
            if not chunk:
                break
            partial_data, valid_image_count = process_chunk(chunk.decode('utf-8'), partial_data, output_dir, valid_image_count)
        receive_end_time = time.time()
        receive_time = receive_end_time - receive_start_time
        print(f"Time taken to receive and process data: {receive_time:.2f} seconds")

if __name__ == "__main__":
    main()
