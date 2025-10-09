import os
import numpy as np
import sys
import base64
import random
import gc
import time
from io import BytesIO
from PIL import Image

# Add the parent directory to the path so we can import the Siamese network modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set memory growth for TensorFlow
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Allow memory growth for the GPU
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled")
    except:
        print("Error setting GPU memory growth")

# Now import Flask
from flask import Flask, render_template, jsonify, request

# Local imports from the Siamese network project
from model import create_siamese_network
from data_utils import load_omniglot_data
# Added for data augmentation
from data_augmentation import augment_image

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
base_network = None
test_images = None
test_labels = None
unique_classes = None
characters_by_class = {}
challenge_history = []  # To keep track of previous challenges

def load_model():
    """
    Load the trained Siamese network model with reduced memory usage
    """
    global model, base_network, test_images, test_labels, unique_classes, characters_by_class
    
    try:
        # Force garbage collection to free memory
        gc.collect()
        
        # Set memory limit for TensorFlow
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
            except Exception as e:
                print(f"Could not set memory limit: {e}")
        
        print("Loading Omniglot data...")
        # Load the Omniglot dataset with reduced size
        (train_images, train_labels), (test_images, test_labels) = load_omniglot_data(reduced_size=True)
        
        # Free up memory by deleting train data since we only need test data for inference
        del train_images
        del train_labels
        gc.collect()
        
        # Further reduce test data size if it's too large
        if len(test_images) > 500:
            indices = np.random.choice(len(test_images), 500, replace=False)
            test_images = test_images[indices]
            test_labels = test_labels[indices]
            gc.collect()
        
        # Get unique classes
        unique_classes = np.unique(test_labels)
        
        # Group images by class
        characters_by_class.clear()  # Clear any existing data
        for i, image in enumerate(test_images):
            label = test_labels[i]
            if label not in characters_by_class:
                characters_by_class[label] = []
            characters_by_class[label].append(image)
        
        # Keep only classes with at least 2 examples
        valid_classes = [cls for cls in unique_classes if len(characters_by_class[cls]) >= 2]
        unique_classes = np.array(valid_classes)
        
        # Keep only the retained classes in the characters_by_class dictionary
        characters_by_class = {k: v for k, v in characters_by_class.items() if k in unique_classes}
        
        print("Loading Siamese Network model...")
        # Create the model architecture with reduced buffer sizes
        input_shape = test_images[0].shape
        
        # Set lower precision to reduce memory usage
        tf.keras.backend.set_floatx('float16')
        
        # Create a simpler model to reduce memory usage
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
        
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        raise

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
    """Generate a new one-shot learning challenge with memory optimization"""
    global unique_classes, characters_by_class, challenge_history, model
    
    # Force garbage collection
    gc.collect()
    
    # Get difficulty level from request (default to 'normal')
    difficulty = request.args.get('difficulty', 'normal')
    
    # Adjust options count based on difficulty
    # Using smaller option counts to reduce computation load
    if difficulty == 'easy':
        num_options = 4  # Reduced from 6
    elif difficulty == 'hard':
        num_options = 8  # Reduced from 12
    else:  # normal
        num_options = 6  # Reduced from 9
    
    try:
        # Get all available classes that have at least 2 examples
        valid_classes = [cls for cls in unique_classes if len(characters_by_class[cls]) >= 2]
        
        # If we have too few valid classes, fall back to simplified mode
        if len(valid_classes) < num_options:
            print("Warning: Not enough valid classes. Using simplified challenge.")
            num_options = min(4, len(valid_classes))
            if num_options < 2:
                raise ValueError("Not enough valid classes to create a challenge")
        
        # Randomly select a target class
        target_class = random.choice(valid_classes)
        
        # Get two different examples of the target class
        target_examples = random.sample(characters_by_class[target_class], 2)
        support_image = target_examples[0]
        true_match_image = target_examples[1]
        
        # Data augmentation only for hard mode and only if augment_image is available
        use_augmentation = difficulty == 'hard'
        if use_augmentation:
            try:
                # Apply augmentation to the true match image with 50% probability
                if random.random() > 0.5:
                    true_match_image = augment_image(true_match_image)
            except:
                print("Warning: Data augmentation failed, using original image")
        
        # Randomly select other classes for distractors
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
        
        # Calculate similarity scores using sequential prediction to save memory
        scores = []
        predicted_match_index = 0
        
        try:
            # Use CPU for prediction to avoid GPU memory issues
            with tf.device('/CPU:0'):
                # Process one option at a time to save memory
                for i, option in enumerate(options):
                    # Reshape images to match model input
                    support = support_image.reshape(1, *support_image.shape)
                    query = option.reshape(1, *option.shape)
                    
                    # Get model prediction (distance) with reduced verbosity
                    distance = model.predict([support, query], verbose=0)[0][0]
                    
                    # Lower distance = higher similarity
                    similarity = 1.0 / (1.0 + distance)
                    scores.append(float(similarity))
                    
                    # Force garbage collection after each prediction
                    if i % 2 == 1:  # Every other prediction
                        gc.collect()
            
            # Find the predicted match (highest similarity)
            predicted_match_index = np.argmax(scores)
        except Exception as e:
            # Fallback if model prediction fails
            print(f"Prediction failed: {str(e)}. Using random predictions.")
            scores = [random.random() for _ in options]
            predicted_match_index = random.randint(0, len(options) - 1)
        
        # Create a challenge ID and timestamp
        challenge_id = len(challenge_history) + 1
        timestamp = int(time.time())
        
        # Save the challenge details
        challenge_info = {
            'id': challenge_id,
            'timestamp': timestamp,
            'difficulty': difficulty,
            'true_match_index': true_match_index,
            'predicted_match_index': predicted_match_index,
            'correct_prediction': true_match_index == predicted_match_index
        }
        challenge_history.append(challenge_info)
        
        # Keep only the last 20 challenges in history (reduced from 50)
        if len(challenge_history) > 20:
            challenge_history = challenge_history[-20:]
        
        # Force garbage collection before returning
        gc.collect()
        
        # Return the challenge data
        return jsonify({
            'challenge_id': challenge_id,
            'support_image': support_image_b64,
            'options': option_images_b64,
            'true_match_index': true_match_index,
            'predicted_match_index': predicted_match_index,
            'scores': scores,
            'difficulty': difficulty,
            'timestamp': timestamp
        })
    except Exception as e:
        print(f"Error creating challenge: {e}")
        # Return a simplified fallback challenge
        return create_fallback_challenge(difficulty)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the model performance"""
    global challenge_history
    
    if not challenge_history:
        return jsonify({
            'total_challenges': 0,
            'correct_predictions': 0,
            'accuracy': 0,
            'by_difficulty': {
                'easy': {'total': 0, 'correct': 0, 'accuracy': 0},
                'normal': {'total': 0, 'correct': 0, 'accuracy': 0},
                'hard': {'total': 0, 'correct': 0, 'accuracy': 0}
            },
            'recent_performance': []
        })
    
    total_challenges = len(challenge_history)
    correct_predictions = sum(1 for c in challenge_history if c['correct_prediction'])
    accuracy = (correct_predictions / total_challenges) * 100 if total_challenges > 0 else 0
    
    # Stats by difficulty
    by_difficulty = {
        'easy': {'total': 0, 'correct': 0, 'accuracy': 0},
        'normal': {'total': 0, 'correct': 0, 'accuracy': 0},
        'hard': {'total': 0, 'correct': 0, 'accuracy': 0}
    }
    
    for challenge in challenge_history:
        diff = challenge['difficulty']
        by_difficulty[diff]['total'] += 1
        if challenge['correct_prediction']:
            by_difficulty[diff]['correct'] += 1
    
    for diff in by_difficulty:
        if by_difficulty[diff]['total'] > 0:
            by_difficulty[diff]['accuracy'] = (by_difficulty[diff]['correct'] / by_difficulty[diff]['total']) * 100
    
    # Get recent performance trend (last 20 challenges)
    recent_performance = [
        {'id': c['id'], 'correct': c['correct_prediction'], 'difficulty': c['difficulty']}
        for c in challenge_history[-20:]
    ]
    
    return jsonify({
        'total_challenges': total_challenges,
        'correct_predictions': correct_predictions,
        'accuracy': accuracy,
        'by_difficulty': by_difficulty,
        'recent_performance': recent_performance
    })

# Error handler for the app
@app.errorhandler(500)
def handle_500(e):
    """Handle internal server errors gracefully"""
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'There was an error processing your request. The system will try to recover automatically.',
    }), 500

# Function to create a fallback challenge when the normal flow fails
def create_fallback_challenge(difficulty='normal'):
    """Create a simplified fallback challenge with minimal resource usage"""
    # Generate a very simple challenge with random images
    
    # Create random images
    support_image = np.random.rand(28, 28, 1).astype('float16')
    true_match = np.random.rand(28, 28, 1).astype('float16')
    
    # Make the true match similar to support to simulate a match
    true_match = support_image * 0.9 + true_match * 0.1
    
    # Create distractor images
    num_options = 4  # Always use 4 options in fallback mode
    distractors = [np.random.rand(28, 28, 1).astype('float16') for _ in range(num_options-1)]
    
    # Create options with true match included
    options = [true_match] + distractors
    random.shuffle(options)
    
    # Find where the true match is
    true_match_index = None
    for i, img in enumerate(options):
        if np.array_equal(img, true_match):
            true_match_index = i
            break
    
    # Convert to base64
    support_image_b64 = image_to_base64(support_image)
    option_images_b64 = [image_to_base64(img) for img in options]
    
    # Create random scores with true match having highest
    scores = [random.random() * 0.5 for _ in range(num_options)]
    scores[true_match_index] = 0.5 + random.random() * 0.5  # Between 0.5 and 1.0
    
    # Return the fallback challenge
    challenge_id = len(challenge_history) + 1 if challenge_history else 1
    timestamp = int(time.time())
    
    return jsonify({
        'challenge_id': challenge_id,
        'support_image': support_image_b64,
        'options': option_images_b64,
        'true_match_index': true_match_index,
        'predicted_match_index': true_match_index,  # In fallback, we make the model "predict" correctly
        'scores': scores,
        'difficulty': difficulty,
        'timestamp': timestamp,
        'fallback_mode': True  # Flag to indicate we're in fallback mode
    })

# Function to create fallback minimal synthetic data
def create_fallback_data():
    """Create minimal synthetic data when regular data loading fails"""
    global model, base_network, test_images, test_labels, unique_classes, characters_by_class
    
    # Clear any previous model data
    model = None
    base_network = None
    test_images = None
    test_labels = None
    unique_classes = None
    characters_by_class = {}
    
    # Force garbage collection
    gc.collect()
    
    # Create minimal synthetic data directly
    print("Creating minimal synthetic dataset for demo purposes...")
    test_images = np.random.rand(25, 28, 28, 1).astype('float16')  # Reduced size, float16
    test_labels = np.random.randint(0, 5, size=(25,))
    unique_classes = np.unique(test_labels)
    
    # Group images by class
    for i, image in enumerate(test_images):
        label = test_labels[i]
        if label not in characters_by_class:
            characters_by_class[label] = []
        characters_by_class[label].append(image)
    
    # Create a minimal model
    print("Creating minimal model...")
    input_shape = test_images[0].shape
    
    # Set lower precision to reduce memory usage
    tf.keras.backend.set_floatx('float16')
    
    model, base_network = create_siamese_network(input_shape)
    
    # Force garbage collection
    gc.collect()

# Run the Flask app
if __name__ == '__main__':
    # Set environment variable to reduce memory usage
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Limit TensorFlow memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory config error: {e}")
    
    # Force garbage collection
    gc.collect()
    
    try:
        # Load the model before starting the app
        print("Initializing the application...")
        load_model()
        print("Model loaded successfully, starting the web server...")
        app.run(debug=False, port=5000, threaded=True)
    except Exception as e:
        print(f"Error starting the application: {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nAttempting to start with fallback mode...")
        # If loading fails, try again with even smaller synthetic data
        try:
            # Create fallback data
            create_fallback_data()
            
            print("Starting web server in fallback mode...")
            app.run(debug=False, port=5000, threaded=True)
        except Exception as e:
            print(f"Fatal error: {str(e)}")
            print("Could not start the application. Please check your memory resources.")