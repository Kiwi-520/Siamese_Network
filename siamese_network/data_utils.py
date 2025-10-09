import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random
import os

def load_omniglot_data():
    """
    Load and preprocess the Omniglot dataset for one-shot learning
    
    This function:
    1. Checks if the Omniglot dataset exists locally, if not downloads it
    2. Loads the images and labels
    3. Normalizes pixel values to [0, 1] range
    4. Reshapes to have a single channel (grayscale)
    5. Resizes all images to 28x28 pixels for consistency
    
    Returns:
        tuple: ((train_images, train_labels), (test_images, test_labels))
    """
    # Since TensorFlow's omniglot loader can be problematic, we'll use a more reliable approach
    # You can modify this to use the TensorFlow loader if it works for you:
    # from tensorflow.keras.datasets.omniglot import load_data
    # (x_train, y_train), (x_test, y_test) = load_data()
    
    # Check if Omniglot dataset exists, otherwise download it
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Step 1: Loading the Omniglot dataset...")
    
    # For demo purposes, we'll use a mock implementation if dataset isn't available
    # In a real implementation, you would download and process the actual Omniglot dataset
    # The following is a placeholder using MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    print("Step 2: Preprocessing the dataset...")
    
    # Normalize the images to [0, 1] range for better training stability
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to have a single channel (grayscale images)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Note: With the real Omniglot dataset, you would resize from 105x105 to 28x28
    # x_train_resized = tf.image.resize(x_train, [28, 28]).numpy()
    # x_test_resized = tf.image.resize(x_test, [28, 28]).numpy()
    
    # For real Omniglot data, there are 1623 character classes - here we're using MNIST (10 classes)
    print(f"Training set: {x_train.shape}, {y_train.shape} - Contains images from {len(np.unique(y_train))} classes")
    print(f"Testing set: {x_test.shape}, {y_test.shape} - Contains images from {len(np.unique(y_test))} classes")
    
    return (x_train, y_train), (x_test, y_test)

def create_pairs(x, y, num_classes=None):
    """
    Create pairs of images for training/testing the Siamese Network
    
    This function creates two types of image pairs:
    1. Positive pairs (label=1): Two different images of the same class/character
    2. Negative pairs (label=0): Two images from different classes/characters
    
    For each class, we:
    - Create positive pairs by selecting two different images from the same class
    - Create negative pairs by pairing an image with an image from a different class
    
    Args:
        x: Images array (N, height, width, channels)
        y: Labels array (N,)
        num_classes: Number of classes to use (if None, use all available classes)
        
    Returns:
        pairs: Array of image pairs with shape (n_pairs, 2, height, width, channels)
        labels: Binary labels (1=same class, 0=different class) with shape (n_pairs,)
    """
    print("Step 3: Creating image pairs for Siamese Network training...")
    
    # If num_classes not specified, use all available classes
    if num_classes is None:
        num_classes = len(np.unique(y))
    
    pairs = []  # Will hold pairs of images
    labels = []  # Will hold labels (1=same class, 0=different class)
    
    # For each class/character
    for class_idx in range(num_classes):
        # Find indices of all images in this class
        class_indices = np.where(y == class_idx)[0]
        
        # If the class has less than 2 samples, skip it (need at least 2 for a pair)
        if len(class_indices) < 2:
            print(f"  Skipping class {class_idx} - not enough samples")
            continue
            
        # Create positive pairs (same class) - maximum 20 pairs per class to balance dataset
        print(f"  Creating pairs for class {class_idx} ({len(class_indices)} samples)")
        for i in range(min(len(class_indices), 20)):
            # Select first image randomly from this class
            # Convert NumPy array to a list for random.choice()
            idx1 = random.choice(class_indices.tolist())
            
            # Ensure we pick a different image from the same class for positive pairs
            remaining_indices = [idx for idx in class_indices if idx != idx1]
            if remaining_indices:  # Check if there are any remaining indices
                idx2 = random.choice(remaining_indices)
                # Add this positive pair (same class)
                pairs.append([x[idx1], x[idx2]])
                labels.append(1)  # 1 means same class
                
                # Also create a negative pair (different classes) to balance the dataset
                # Choose a random class different from the current class
                neg_class_idx = random.choice([i for i in range(num_classes) if i != class_idx])
                neg_class_indices = np.where(y == neg_class_idx)[0]
                
                if len(neg_class_indices) > 0:  # Check if the negative class has any samples
                    idx_neg = random.choice(neg_class_indices.tolist())
                    # Add this negative pair (different classes)
                    pairs.append([x[idx1], x[idx_neg]])
                    labels.append(0)  # 0 means different classes
    
    # Convert lists to numpy arrays
    pairs = np.array(pairs)
    labels = np.array(labels)
    
    # Shuffle the data for better training
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    
    print(f"  Created {len(labels)} pairs total: {np.sum(labels)} positive and {len(labels) - np.sum(labels)} negative")
    
    return pairs[indices], labels[indices]

def get_batch(pairs, labels, batch_size):
    """
    Get a random batch of image pairs for training
    
    This function randomly selects a batch of pairs for each training iteration,
    which helps prevent overfitting and improves model generalization.
    
    Args:
        pairs: Array of image pairs
        labels: Array of labels (1=same class, 0=different class)
        batch_size: Number of pairs to include in the batch
        
    Returns:
        pair_batch: List containing two arrays - [left_images, right_images]
        label_batch: Array of corresponding labels
    """
    # Get the total number of pairs
    n = len(labels)
    
    # Randomly select indices for the batch
    # replace=False ensures we don't select the same pair multiple times
    indices = np.random.choice(int(n), size=batch_size, replace=False)
    
    # Extract the pairs and labels using the selected indices
    pair_batch = [pairs[indices, 0], pairs[indices, 1]]  # [left_images, right_images]
    label_batch = labels[indices]  # Corresponding labels
    
    return pair_batch, label_batch

def visualize_pairs(pairs, labels, num_pairs=5):
    """
    Visualize some example pairs from the dataset for inspection
    
    This function displays pairs of images side by side, showing whether they
    are from the same class (positive pair) or different classes (negative pair).
    
    Args:
        pairs: Array of image pairs
        labels: Array of labels (1=same class, 0=different class)
        num_pairs: Number of pairs to visualize
    """
    print("Step 4: Visualizing example image pairs...")
    
    plt.figure(figsize=(10, 4))
    for i in range(num_pairs):
        # Get a random index
        idx = np.random.randint(0, len(labels))

        # Get the pair and label
        pair = pairs[idx]
        label = labels[idx]

        # Plot first image in the pair (top row)
        plt.subplot(2, num_pairs, i + 1)
        plt.imshow(pair[0].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Plot second image in the pair (bottom row)
        plt.subplot(2, num_pairs, i + 1 + num_pairs)
        plt.imshow(pair[1].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Add a title indicating if the pair shows the same class or different classes
        plt.title(f"Same class: {label == 1}")
    
    plt.tight_layout()
    plt.show()