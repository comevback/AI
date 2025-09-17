import struct
import numpy as np


def load_idx_images(path):
    with open(path, 'rb') as f:

        raw_data = f.read()
        with open("./raw_mnist.txt", "w") as rwf:
            rwf.write(raw_data.hex())

        f.seek(0)

        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number mismatch, got {magic}"

        print("magic: ", magic)
        print("num: ", num)
        print("rows: ", rows)
        print("cols: ", cols)

        data = np.frombuffer(f.read(), dtype=np.uint8)

        # flat the data to N * 784 matrix
        images = data.reshape(num, rows*cols)

        with open("./mnist.txt", "w") as wf:
            wf.write(f"magic: {magic}\n")
            wf.write(f"num: {num}\n")
            wf.write(f"rows: {rows}\n")
            wf.write(f"cols: {cols}\n")
            wf.write(f"images: {images}")

        return images


X_test = load_idx_images("./t10k-images.idx3-ubyte")
print(X_test)
