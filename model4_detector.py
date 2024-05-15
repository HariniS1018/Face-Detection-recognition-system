from werkzeug.utils import secure_filename
from flask import Flask
import cv2, os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Function to load dataset and preprocess images
def load_dataset(dataset_path):
    images = []
    labels = []

    try:
        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            label = int(folder_name)
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        images.append(image)
                        labels.append(label)
                    else:
                        print("Error loading image:", image_path)
    except Exception as e:
        print("Error loading dataset:", e)

    # Length of a list
    list_length = len(images)
    print("Length of the list:", list_length)

    return images, labels

# Load face detection model
def load_face_detection_model():
    prototxt_file = 'deploy.prototxt'
    model_file = 'res10_300x300_ssd_iter_140000.caffemodel'
    
    net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)
    if net.empty():
        print("Error: Unable to load face detection model.")
        return None
    return net

# Detect faces in the image
def detect_faces(image, net):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    detected_faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append((startX, startY, endX, endY))
    return detected_faces

# Draw bounding boxes around detected faces and return coordinates
def draw_boxes(image, boxes):
    face_coordinates = []
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        face_coordinates.append((startX, startY, endX, endY))
    return image, face_coordinates

#Function to detect faces
def detection_model():
    images, labels = load_dataset(dataset_path=r"C:\Users\Admin\OneDrive\Documents\Harini_S\projects\FDS_FRS\photos")
    
    for image, user_id in zip(images, labels):
        if image is None:
            return {'message': 'cannot load the image'}
        
        # Load face detection model
        net = load_face_detection_model()
        if net is None:
            return {'message': 'face detection model cannot be loaded'}, 400
        
        # Detect faces
        detected_faces = detect_faces(image, net)
        
        # Draw bounding boxes
        image_with_boxes, face_coordinates = draw_boxes(image.copy(), detected_faces)

        # Generate filename using user_id
        filename = secure_filename(f"{user_id}.jpg")  # You can use any file extension here
        
        # Save the detected image with user_id as filename
        detected_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(detected_file, image_with_boxes)

    print("face detection model done successfully.")

    return net

def test_detection_model(face_detector, new_data):
    image = cv2.imread(new_data)

    if image is None:
        return {'message': 'cannot load the image'}
    
    # Detect faces
    detected_faces = detect_faces(image, face_detector)
    
    # Draw bounding boxes
    image_with_boxes, face_coordinates = draw_boxes(image.copy(), detected_faces)

    # Display the image with bounding boxes
    print(face_coordinates)
    cv2.imshow("Detected Faces", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return image_with_boxes, face_coordinates

if __name__ == "__main__":
    app.run(port=5001,debug=True)

