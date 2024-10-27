import socket
import base64
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
            return image
        except (base64.binascii.Error, IOError):
            print(f"Skipping invalid base64 image: {b64_image[:30]}...")
            return None
    return None

def process_chunk(chunk, partial_data, valid_images):
    data = partial_data + chunk
    base64_images = data.split("\n")
    
    # Keep the last part of the data as partial data if it might be incomplete
    partial_data = base64_images.pop() if data[-1] != "\n" else ""

    for b64_image in base64_images:
        image = decode_image(b64_image)
        if image:
            valid_images.append(image)
        if len(valid_images) >= 12:
            break

    return partial_data

def main():
    server_address = "54.79.244.247"
    port = 9875

    # Create a TCP/IP socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Measure time taken to connect to the server
        connect_start_time = time.time()
        sock.connect((server_address, port))
        connect_end_time = time.time()
        print(f"Connected to {server_address} on port {port}")
        connect_time = connect_end_time - connect_start_time
        print(f"Time taken to connect: {connect_time:.2f} seconds")

        valid_images = []
        partial_data = ""

        # Measure time taken to receive and process data from the server
        receive_start_time = time.time()
        while len(valid_images) < 12:
            chunk = sock.recv(4096)
            if not chunk:
                break
            partial_data = process_chunk(chunk.decode('utf-8'), partial_data, valid_images)
        receive_end_time = time.time()
        receive_time = receive_end_time - receive_start_time
        print(f"Time taken to receive and process data: {receive_time:.2f} seconds")

        # Access the images in-memory
        for i, image in enumerate(valid_images, start=1):
            print(f"Image {i} in memory: {image}")

if __name__ == "__main__":
    main()
