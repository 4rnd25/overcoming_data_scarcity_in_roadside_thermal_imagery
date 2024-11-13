"""
Created on 11 04 2024 16:00

@author: ISAC - pettirsch
"""

import onnxruntime as ort
import numpy as np
import time

# Parameter
providers = ['CUDAExecutionProvider']
model_path = "*.onnx"
input_size = 640


def speedtest(model_path, providers, input_size):
    # Create onnxruntime session
    sess = ort.InferenceSession(model_path, providers=providers)

    # Create random input tensor
    input_tensor = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

    for i in range(1000):
        if i == 10:
            start = time.time()
        # Perform your model inference here
        outputs = sess.run(None, {"images": input_tensor})
    end = time.time()

    elapsed_time = end - start
    mean_time = elapsed_time / 990

    return mean_time


if __name__ == "__main__":
    mean_time = speedtest(model_path, providers, input_size)
    print(f"Mean inference time: {mean_time} s")