import os
import numpy as np
import tensorflow as tf
import random
import sys
import base64
from io import BytesIO
from PIL import Image

# Make sure we're using the compatible Werkzeug version
import werkzeug
print(f"Using Werkzeug version: {werkzeug.__version__}")

# Add the parent directory to the path so we can import the Siamese network modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import Flask after setting up the path
from flask import Flask, render_template, jsonify, request

# Local imports from the Siamese network project
from model import create_siamese_network
from data_utils import load_omniglot_data

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
base_network = None
test_images = None
test_labels = None
unique_classes = None
characters_by_class = {}

def load_model():
    """
    Load the trained Siamese network model
    """
    global model, base_network, test_images, test_labels, unique_classes, characters_by_class
    
    print("Loading Omniglot data...")
    # Load the Omniglot dataset
    (train_images, train_labels), (test_images, test_labels) = load_omniglot_data()
    
    # Get unique classes
    unique_classes = np.unique(test_labels)
    
    # Group images by class
    for i, image in enumerate(test_images):
        label = test_labels[i]
        if label not in characters_by_class:
            characters_by_class[label] = []
        characters_by_class[label].append(image)
    
    print("Loading Siamese Network model...")
    # Create the model architecture
    input_shape = test_images[0].shape
    model, base_network = create_siamese_network(input_shape)
    
    # Load the weights
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'checkpoints', 'siamese_network_best.h5')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_weights(checkpoint_path)
    else:
        print(f"No saved model found at {checkpoint_path}. Using untrained model.")
    
    print("Model loaded successfully.")

def image_to_base64(img_array):
    """Convert a numpy array image to base64 for HTML display"""
    # Rescale from [0,1] to [0,255]
    img_array = (img_array * 255).astype(np.uint8)
    
    # Reshape to 28x28
    img_array = img_array.reshape(28, 28)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Save to BytesIO object
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    
    # Encode as base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/new-challenge', methods=['GET'])
def new_challenge():
    """Generate a new one-shot learning challenge"""
    global unique_classes, characters_by_class
    
    # Number of options to display
    num_options = 9
    
    # Get all available classes that have at least 2 examples
    valid_classes = [cls for cls in unique_classes if len(characters_by_class[cls]) >= 2]
    
    # Randomly select a target class
    target_class = random.choice(valid_classes)
    
    # Get two different examples of the target class
    target_examples = random.sample(characters_by_class[target_class], 2)
    support_image = target_examples[0]
    true_match_image = target_examples[1]
    
    # Randomly select other classes for distractors
    # Make sure we have enough unique classes
    num_distractors = min(num_options - 1, len(valid_classes) - 1)
    distractor_classes = random.sample([cls for cls in valid_classes if cls != target_class], num_distractors)
    
    # Get one random example from each distractor class
    distractor_images = [random.choice(characters_by_class[cls]) for cls in distractor_classes]
    
    # Create the options (shuffle the true match with distractors)
    options = [true_match_image] + distractor_images
    random.shuffle(options)
    
    # Find where the true match is after shuffling
    true_match_index = None
    for i, img in enumerate(options):
        if np.array_equal(img, true_match_image):
            true_match_index = i
            break
    
    # Convert images to base64 for sending to the frontend
    support_image_b64 = image_to_base64(support_image)
    option_images_b64 = [image_to_base64(img) for img in options]
    
    # Calculate similarity scores
    scores = []
    for option in options:
        # Reshape images to match model input
        support = support_image.reshape(1, *support_image.shape)
        query = option.reshape(1, *option.shape)
        
        # Get model prediction (distance)
        distance = model.predict([support, query])[0][0]
        
        # Lower distance = higher similarity
        similarity = 1.0 / (1.0 + distance)
        scores.append(float(similarity))
    
    # Find the predicted match (lowest distance = highest similarity)
    predicted_match_index = np.argmax(scores)
    
    # Return the challenge data
    return jsonify({
        'support_image': support_image_b64,
        'options': option_images_b64,
        'true_match_index': true_match_index,
        'predicted_match_index': predicted_match_index,
        'scores': scores
    })

# Run the Flask app
if __name__ == '__main__':
    try:
        # Load the model before starting the app
        print("Initializing the application...")
        load_model()
        print("Model loaded successfully, starting the web server...")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        import traceback
        traceback.print_exc()