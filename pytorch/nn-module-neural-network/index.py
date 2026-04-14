import struct
import torch
from torch.utils.data import DataLoader




def extract_images(file_path):
    with open(file_path,'rb') as file:
        magic_number = struct.unpack('>I',file.read(4))[0]
        num_images = struct.unpack('>I',file.read(4))[0]
        num_rows = struct.unpack('>I',file.read(4))[0]
        num_cols = struct.unpack('>I',file.read(4))[0]

        images = []
        for i in range(num_images):
            image = []
            for _ in range(num_rows):
                row = []
                for _ in range(num_cols):
                    pixel = struct.unpack('>B',file.read(1))[0]
                    row.append(pixel)
                image.append(row)
            images.append(image)
        
        return images

def extract_labels(file_path):
    with open(file_path,'rb') as file:
        magic_nuber = struct.unpack('>I',file.read(4))[0]
        num_labels = struct.unpack('>I',file.read(4))[0]
        
        labels = []

        for i in range(num_labels):
            label = struct.unpack('>B',file.read(1))[0]
            labels.append(label)
        return labels
    


data = extract_images("../data/t10k-images-idx3-ubyte")
labels = extract_labels("../data/t10k-labels-idx1-ubyte")
print(len(data))
print(len(labels))
# Path