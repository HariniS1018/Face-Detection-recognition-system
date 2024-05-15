import cv2, os, io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from model3_detection import detection_model, test_detection_model

def preprocess_image(image, target_size=(200, 200)):
    try:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray_image",gray_image)
        
        # Resize image
        resized_image = cv2.resize(gray_image, target_size)
        # cv2.imshow("resized_image",resized_image)
        
        # Histogram Equalization
        equalized_image = cv2.equalizeHist(resized_image)
        # cv2.imshow("equalized_image",equalized_image)
        
        # Normalization
        normalized_image = equalized_image / 255.0
        # cv2.imshow("normalized_image",normalized_image)
        
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        return normalized_image
    
    except Exception as e:
        print("Error during preprocessing:", e)
        return None

# Function to load dataset and preprocess images
def load_dataset(dataset_path):
    images = []
    labels = []

    try:
        for folder_name in os.listdir(dataset_path):        # to Iterate the list of all files and directories in the specified directory.
            # print(dataset_path)
            folder_path = os.path.join(dataset_path, folder_name)
            # print(folder_name + "  " + folder_path)
            if not os.path.isdir(folder_path):          # if the expected structure of the dataset is not a directory, meant folder of images of an individual, skip it.
                continue
            label = int(folder_name)  # Assuming folder names are numeric labels
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        preprocessed_image = preprocess_image(image)
                        if preprocessed_image is not None:
                            images.append(preprocessed_image)
                            labels.append(label)
                        else:
                            print("Error preprocessing image as the above exception says in the image: ", image_path)
                    else:
                        print("Error loading image:", image_path)
    except Exception as e:
        print("Error loading dataset:", e)

    # Length of a list
    list_length = len(images)
    print("Length of the list:", list_length)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Function to split dataset into train and test sets
def split_dataset(images, labels, test_size=0.2, random_state=42):
    return train_test_split(images, labels, test_size=test_size, random_state=random_state)

def train_face_recognition(X_train, y_train, save_model_path='face_recognition_model.xml'):
    # Create LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Train the face recognizer
    face_recognizer.train(X_train, np.array(y_train))

    # Save the trained face recognizer model if path is provided
    if save_model_path:
        face_recognizer.save(save_model_path)
        print("Trained face recognizer model saved successfully.")

    return face_recognizer

# Function to test face recognition model
def test_face_recognition(face_recognizer, X_test):
    predicted_labels = []  # List to store the predicted labels

    # Iterate over test images
    for image in X_test:
        # Use the trained face recognizer to recognize faces in the current image
        predicted_label, _ = face_recognizer.predict(image)

        # Append the predicted label to the list
        predicted_labels.append(predicted_label)

    return predicted_labels

# Function to evaluate face recognition model
def evaluate_recognition_model(y_pred, y_test):
    # Calculate accuracy
    recognition_accuracy = np.mean(np.array(y_test) == np.array(y_pred))

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    return recognition_accuracy, conf_matrix

def display_model_performance(accuracy, cm):
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(cm)

def detect_and_recognize_faces(face_detector, face_recognizer, new_data):
    # Perform face detection using the face_detector
    detected_faces, face_coordinates = test_detection_model(face_detector, new_data)
    
    # Initialize list to store recognized identities
    recognized_identities = []
    
    # Iterate over detected faces
    # for face in detected_faces:
    try:
        for tuple_value in face_coordinates:
            x, y, w, h = tuple_value
        preprocessed_image = preprocess_image(detected_faces)
        x, y, w, h = int(x), int(y), int(w), int(h)
        face_image = preprocessed_image[y:y+h, x:x+w]
        cv2.imshow("Detected Faces", face_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print dimensions of the face image
        print("Dimensions of face image:", face_image.shape)
        
        # Perform face recognition using the face_recognizer
        identity, _ = face_recognizer.predict(face_image)

        # Add the recognized identity to the list
        recognized_identities.append(identity)

    except Exception as e:
        print("Error processing face:", e)


    return recognized_identities

# -------------------------------------------------------------------------------------------------------------------------------------

# Main function
def main():
    # Step 1: Load and preprocess dataset
    dataset_path = r"C:\Users\Admin\OneDrive\Documents\Harini_S\projects\FDS_FRS\photos"
    images, labels = load_dataset(dataset_path)

    # Step 2: Split dataset into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(images, labels)

    # Step 3: Call face detection model
    face_detector = detection_model()

    # Step 4: Train face recognition model
    face_recognizer = train_face_recognition(X_train, y_train)

    # Step 5: Test face recognition model
    y_pred = test_face_recognition(face_recognizer, X_test)

    # Step 6: Model evaluation
    recognition_accuracy, recognizer_conf_matrix = evaluate_recognition_model(y_pred, y_test)
    display_model_performance(recognition_accuracy, recognizer_conf_matrix)

    return 200

def load_face_recognizer():
    folder_path = "C:/Users/Admin/OneDrive/Documents/Harini_S/projects/FDS_FRS/face_recognition_model.xml"
    
    # Construct the path to the XML file
    recognizer_file = os.path.join(folder_path, 'face_recognizer.xml')

    # Load the face recognizer from the XML file
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(folder_path)

    return face_recognizer

def test(image_bytes):
    image_file = io.BytesIO(image_bytes)
    image_data = np.frombuffer(image_file.read(), np.uint8)

    # Decode the image data
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # image = load_image(image)
    if image is None:
        return {'message': 'cannot load the image'}
    
    face_recognizer = load_face_recognizer()
    face_detector = detection_model()
    # image_path = r"C:\Users\Admin\OneDrive\Documents\Harini_S\projects\FDS_FRS\photos\9001\a.jpg"
    user_ids = detect_and_recognize_faces(face_detector,face_recognizer, image)
    return user_ids, 200

if __name__ == "__main__":
    main()
