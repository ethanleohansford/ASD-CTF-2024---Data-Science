import socket
import time

def check_connection_duration(host: str, port: int) -> float:
    start_time = time.time()
    
    try:
        with socket.create_connection((host, port)) as sock:
            print(f"Connected to {host} on port {port}")
            while True:
                try:
                    # Try to receive data from the server
                    data = sock.recv(1024)
                    if not data:
                        # No data means the connection is closed
                        break
                except socket.error:
                    break
    except socket.error as e:
        print(f"Failed to connect: {e}")
        return 0.0
    
    end_time = time.time()
    duration = end_time - start_time
    return duration

host = "54.79.244.247"
port = 9875

duration = check_connection_duration(host, port)
if duration > 0:
    print(f"Connection lasted for {duration:.2f} seconds.")
else:
    print("Could not establish connection.")
