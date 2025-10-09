import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras import backend as K

def euclidean_distance(vectors):
    """
    Calculate the Euclidean distance between two feature vectors
    
    The Euclidean distance is the "straight-line" distance between two points
    in Euclidean space. For feature vectors, it represents how similar two
    inputs are - smaller distances mean more similar inputs.
    
    With L2 normalized vectors, this distance is proportional to cosine similarity:
    d(x,y)^2 = ||x||^2 + ||y||^2 - 2*x·y = 2 - 2*cos(x,y) when ||x||=||y||=1
    
    Mathematical formula:
    distance = sqrt(sum((x_i - y_i)^2))
    
    Args:
        vectors: List containing two tensors [x, y] of the same length
        
    Returns:
        Euclidean distance tensor
    """
    # Unpack the two vectors
    x, y = vectors
    
    # For L2 normalized vectors, we can use a simplified distance calculation
    # This is more stable and works well with normalized embeddings
    # For non-normalized vectors, this is equivalent to the original formula
    
    # Calculate sum of squared differences
    # K.square(x - y) computes element-wise squared difference
    # We sum across all features (axis=1) and keep dimensions
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    
    # Take square root of the sum
    # K.epsilon() is added to prevent numerical instability from sqrt(0)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss function for Siamese Networks
    
    This custom loss function has two components:
    1. For similar pairs (y_true=1): Loss = distance²
       - This penalizes the network for predicting large distances between similar inputs
       - Minimizing this pulls similar points closer together in the embedding space
       
    2. For dissimilar pairs (y_true=0): Loss = max(0, margin - distance)²
       - This only penalizes if distance < margin
       - Ensures dissimilar pairs are at least 'margin' distance apart
       - Minimizing this pushes dissimilar points further apart
    
    Mathematical formula:
    L(y,d) = y * d² + (1-y) * max(0, margin - d)²
    
    where:
    - y is the true label (1 for similar, 0 for dissimilar)
    - d is the predicted distance
    - margin is the minimum distance we want between dissimilar pairs
    
    Args:
        y_true: True binary labels: 1 for similar pairs, 0 for dissimilar
        y_pred: Euclidean distance between pairs from the model
        
    Returns:
        Contrastive loss value (scalar)
    """
    # Margin parameter defines how far apart dissimilar pairs should be
    # Increased from 1.0 to 2.0 to create more separation between classes
    # Dissimilar pairs with distance < margin will contribute to the loss
    margin = 2.0
    
    # Component 1: For similar pairs (y_true=1), penalize distance
    # Square the distance for similar pairs
    square_pred = K.square(y_pred)
    
    # Component 2: For dissimilar pairs (y_true=0), penalize if distance < margin
    # Only contribute to loss if distance is less than margin
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    
    # Combine both components
    # - When y_true=1 (similar): first term is 0, second term is square_pred
    # - When y_true=0 (dissimilar): first term is margin_square, second term is 0
    loss = K.mean((1 - y_true) * margin_square + y_true * square_pred)
    
    return loss

def create_base_network(input_shape):
    """
    Create the improved base CNN for feature extraction
    
    This function defines an enhanced CNN architecture with batch normalization 
    and deeper convolutional layers for better feature extraction.
    
    Architecture:
    1. Input image
    2. Four convolutional blocks, each with:
       - Conv2D layer to extract features
       - BatchNormalization to stabilize learning
       - ReLU activation
       - MaxPooling to reduce dimensionality
    3. Flatten layer to convert 2D feature maps to 1D vector
    4. Two dense layers with dropout for the final embedding
    
    Args:
        input_shape: Shape of the input image (height, width, channels)
        
    Returns:
        Base network model that converts an image to a feature embedding
    """
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # First convolutional block with batch normalization
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Second convolutional block with batch normalization
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Third convolutional block with batch normalization
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Fourth convolutional block with batch normalization (new)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten the 2D feature maps to a 1D feature vector
    x = Flatten()(x)
    
    # Hidden dense layer with dropout
    x = Dense(256, activation=None)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Final dense layer to produce the embedding
    embedding = Dense(128, activation=None)(x)
    embedding = BatchNormalization()(embedding)
    embedding = tf.keras.layers.Activation('relu')(embedding)
    
    # Add L2 normalization to standardize the embeddings
    # This helps ensure that the distance calculations are more consistent
    embedding = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.l2_normalize(x, axis=1)
    )(embedding)
    
    # Create the model
    return Model(input_layer, embedding, name="SiameseBaseNetwork")

def create_siamese_network(input_shape):
    """
    Create the complete Siamese Network architecture
    
    A Siamese Network consists of:
    1. Two identical CNN branches with shared weights (the base network)
    2. Each branch processes one input image to produce an embedding
    3. A distance function that compares the two embeddings
    
    The key insight is that the same exact network (with the same weights)
    processes both images, ensuring they're mapped to the same embedding space.
    
    Args:
        input_shape: Shape of the input image (height, width, channels)
        
    Returns:
        model: Complete Siamese network model
        base_network: The base CNN used in each branch (for later visualization/analysis)
    """
    print("Step 5: Creating the Siamese Network architecture...")
    
    # Define the inputs for the two images to be compared
    left_input = Input(shape=input_shape, name="LeftImageInput")
    right_input = Input(shape=input_shape, name="RightImageInput")
    
    # Create the base network - this is the CNN that will process each image
    # Both branches of the Siamese network will share the same weights
    print("  Creating base CNN architecture...")
    base_network = create_base_network(input_shape)
    
    # Process both inputs through the same base network
    # This ensures both images are processed identically
    print("  Creating twin branches with shared weights...")
    left_features = base_network(left_input)   # Process the left image
    right_features = base_network(right_input) # Process the right image with the same network
    
    # Calculate the Euclidean distance between the two feature vectors
    # This measures how similar the two inputs are
    print("  Adding distance function to compare embeddings...")
    distance = Lambda(euclidean_distance, name="EuclideanDistance")([left_features, right_features])
    
    # Create the complete model
    # Inputs: Two images
    # Output: Distance value (smaller = more similar)
    model = Model(inputs=[left_input, right_input], outputs=distance, name="SiameseNetwork")
    
    # Compile the model with our custom contrastive loss and the Adam optimizer
    print("  Compiling model with contrastive loss...")
    model.compile(loss=contrastive_loss, optimizer='adam', metrics=['accuracy'])
    
    return model, base_network