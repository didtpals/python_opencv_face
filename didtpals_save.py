import cv2
import os
import numpy as np

# Define the path to the dataset
datafile_path = "didtpals_img"

datafile = "./didtpals/training_file.yml"

# Create the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

known_faces = {}
i = 0
for name in os.listdir(datafile_path):
    i += 1

    known_faces[name] = i


# known_faces 안에는 사진 파일 이름과 그에 대응하는 특정 값이 있어야함 

# Create the training data
faces = []
labels = []
for name in os.listdir(datafile_path):
    image = cv2.imread(f"{datafile_path}/{name}", cv2.IMREAD_GRAYSCALE)
    label = known_faces[name]
    faces.append(image)
    labels.append(label)
    
    print(name + '파일 학습')

# Train the face recognizer
recognizer.train(faces, np.array(labels))

# Save the face model
recognizer.save(datafile)