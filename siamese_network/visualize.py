import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_distance_distribution(predictions, labels):
    """
    Plot the distribution of distances for similar and dissimilar pairs
    
    Args:
        predictions: Distance predictions from the model
        labels: True labels
    """
    similar_dist = predictions[labels == 1]
    dissimilar_dist = predictions[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(similar_dist, bins=50, alpha=0.5, label='Similar pairs')
    plt.hist(dissimilar_dist, bins=50, alpha=0.5, label='Dissimilar pairs')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution for Similar and Dissimilar Pairs')
    plt.legend()
    plt.show()

def visualize_embeddings(base_network, images, labels):
    """
    Visualize embeddings using t-SNE
    
    Args:
        base_network: Base network model for feature extraction
        images: Images to visualize
        labels: Labels for the images
    """
    # Get the actual number of samples available
    num_samples = len(images)
    
    # If we have enough samples, take a random subset
    if num_samples > 100:
        # Use a smaller sample size to ensure t-SNE runs efficiently
        n_samples = min(400, num_samples)
        indices = np.random.choice(range(num_samples), size=n_samples, replace=False)
        sampled_images = images[indices]
        sampled_labels = labels[indices]
    else:
        # Use all available samples if we don't have many
        sampled_images = images
        sampled_labels = labels
    
    # Get embeddings from the base network
    embeddings = base_network.predict(sampled_images)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot the embeddings
    plt.figure(figsize=(10, 8))
    
    # Get unique labels
    unique_labels = np.unique(sampled_labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = sampled_labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], c=[colors[i]], label=f'Class {label}', alpha=0.7)
    
    plt.title('t-SNE Visualization of Embeddings')
    plt.legend()
    plt.show()

def visualize_predictions(model, test_pairs, test_labels, threshold=0.5, num_examples=5):
    """
    Visualize some predictions of the Siamese Network
    
    Args:
        model: Trained Siamese Network model
        test_pairs: Test pairs
        test_labels: Test labels
        threshold: Distance threshold for similar/dissimilar classification
        num_examples: Number of examples to visualize
    """
    # Get a random subset of test examples
    # Convert to int to avoid any potential numpy array issues
    indices = np.random.choice(int(len(test_labels)), size=num_examples, replace=False)
    
    # Get the selected pairs and labels
    selected_pairs = test_pairs[indices]
    selected_labels = test_labels[indices]
    
    # Make predictions
    left_images = selected_pairs[:, 0]
    right_images = selected_pairs[:, 1]
    predictions = model.predict([left_images, right_images])
    
    # Convert distances to binary predictions
    binary_predictions = np.where(predictions < threshold, 1, 0)
    
    # Visualize the examples
    plt.figure(figsize=(15, 3*num_examples))
    for i in range(num_examples):
        # Plot left image
        plt.subplot(num_examples, 3, i*3 + 1)
        plt.imshow(left_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Plot right image
        plt.subplot(num_examples, 3, i*3 + 2)
        plt.imshow(right_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        
        # Plot prediction
        plt.subplot(num_examples, 3, i*3 + 3)
        plt.text(0.5, 0.5, f'True: {"Same" if selected_labels[i] == 1 else "Different"}\n'
                           f'Pred: {"Same" if binary_predictions[i] == 1 else "Different"}\n'
                           f'Distance: {predictions[i][0]:.3f}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()