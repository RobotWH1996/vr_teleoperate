import cv2
import zmq
import pickle
import zlib
import numpy as np


def start_server():
    # camera init
    cap = cv2.VideoCapture(4, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # set ZeroMQ context and socket
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind("tcp://*:5555")
    print("The server has started, waiting for client connections...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame read is error")
            break
        h, w, c = frame.shape
        print(f"Captured frame resolution: {w}x{h}, channels: {c}")
        # encoding image

        frame_expanded = np.hstack((frame, frame))
        h, w, c = frame_expanded.shape
        # print(f"Captured frame_expanded resolution: {w}x{h}, channels: {c}")
        ret2, frame_expanded = cv2.imencode(".jpg", frame_expanded)

        if not ret2:
            continue
        # Compressing data using pickle and zlib
        data = pickle.dumps(frame_expanded)
        compressed_data = zlib.compress(data)

        # sending data in pieces
        chunk_size = 60000
        num_chunks = len(compressed_data) // chunk_size + 1
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = compressed_data[start:end]
            socket.send(chunk)
        print("send picture")
    cap.release()
    context.term()


if __name__ == "__main__":
    start_server()
