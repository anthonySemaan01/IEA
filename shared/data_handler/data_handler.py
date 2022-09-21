from PIL import Image  # PIL is a library used to read / write images-ubyte
from domain.models.file_structure import FileStructure
import numpy as np

DATASET = ''  # 'mnist' or 'extended-mnist' or 'extended-mnist-letters'
DATA_DIR = ''
TRAIN_DATA_FILENAME = "../../datasets/training/images-ubyte/emnist-letters-train-images-idx3-ubyte"


def read_image(path):
    return np.asarray(Image.open(path).convert('L'))


def write_image(image, path):
    img = Image.fromarray(np.array(image), 'L')
    img.save(path)


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):  # we are reading images-ubyte pixel by pixel
    # n_max_images specifies how many sample we want from the file
    images = []
    with open(filename, 'rb') as f:
        _ = bytes_to_int(f.read(4))  # 4 because each sample is of 32 bits == 4 bytes
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


if __name__ == '__main__':
    X_train = read_images(TRAIN_DATA_FILENAME, 10000)
    for idx, test_sample in enumerate(X_train):
        write_image(test_sample, "../../datasets/training/images" + f"/{idx}.png")
