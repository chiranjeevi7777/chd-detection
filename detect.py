import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Configuration (update these paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'inceptionv3_model2.h5')  # Path to your saved model
CLASS_NAMES = ['ASD', 'Normal', 'PDA', 'VSD']  # Must match training order
IMG_SIZE = (299, 299)  # Must match training size (InceptionV3 uses 299x299)

def load_model(model_path):
    """Load the pre-trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    """Preprocess the image for model prediction"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    return img_array, img

def predict_chd(model, img_array):
    """Make prediction and return class probabilities"""
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence, predictions[0]

def display_result(img, prediction, confidence, probabilities):
    """Display the image with prediction results"""
    plt.figure(figsize=(10, 6))
    
    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {prediction}\nConfidence: {confidence:.2f}%")
    plt.axis('off')
    
    # Show probabilities
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(CLASS_NAMES))
    plt.barh(y_pos, probabilities, align='center')
    plt.yticks(y_pos, CLASS_NAMES)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load the model
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Get image path from user
    img_path = os.path.join(BASE_DIR, 'inputs', 'xray', 'PDA01.jpg')
    
    try:
        # Preprocess and predict
        img_array, original_img = preprocess_image(img_path)
        predicted_class, confidence, probabilities = predict_chd(model, img_array)
        
        # Display results
        print(f"\nPrediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("\nClass Probabilities:")
        for class_name, prob in zip(CLASS_NAMES, probabilities):
            print(f"{class_name}: {prob*100:.2f}%")
        
        display_result(original_img, predicted_class, confidence, probabilities)
    
    except Exception as e:
        print(f"\nError processing image: {e}")

if __name__ == "__main__":
    main()