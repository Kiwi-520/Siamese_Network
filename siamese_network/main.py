import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

# Local imports
from data_utils import load_omniglot_data, create_pairs, visualize_pairs
from model import create_siamese_network
from train import train_siamese_network, plot_training_history
from evaluate import evaluate_model, plot_roc_curve, one_shot_test
from visualize import plot_distance_distribution, visualize_embeddings, visualize_predictions

# Set random seeds for reproducibility
# This ensures we get the same results each time we run the code
print("Setting up the environment...")
np.random.seed(42)
tf.random.set_seed(42)

# Make sure TensorFlow uses a reasonable amount of GPU memory
# This prevents TensorFlow from allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU acceleration enabled: {len(gpus)} GPU(s) available")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

def main():
    """
    Main function to run the entire Siamese Network pipeline
    
    This function orchestrates the complete workflow:
    1. Data loading and preprocessing
    2. Creating image pairs for training
    3. Building the Siamese Network model
    4. Training the model
    5. Evaluating performance
    6. Visualizing results and embeddings
    7. Testing one-shot learning capability
    """
    print("\n" + "="*80)
    print(" SIAMESE NETWORK FOR ONE-SHOT LEARNING ".center(80, "="))
    print("="*80 + "\n")
    
    # STEP 1: Load and preprocess the Omniglot dataset
    (train_images, train_labels), (test_images, test_labels) = load_omniglot_data()
    
    # STEP 2-3: Create pairs for training and testing
    train_pairs, train_labels = create_pairs(train_images, train_labels)
    test_pairs, test_labels = create_pairs(test_images, test_labels)
    
    # STEP 4: Visualize some example pairs
    visualize_pairs(train_pairs, train_labels)
    
    # STEP 5: Create the Siamese Network
    input_shape = train_images[0].shape
    model, base_network = create_siamese_network(input_shape)
    
    print("\nSiamese Network Architecture Summary:")
    model.summary()
    
    # Create a checkpoint to save the best model during training
    print("\nSetting up model checkpointing...")
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'siamese_network_best.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    # STEP 6: Train the model with improved settings
    # Increased epochs and batch_size for better training
    # Added callbacks for early stopping and model checkpointing
    from tensorflow.keras.callbacks import EarlyStopping
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Use the maximum amount of validation data available
    max_val_samples = min(1000, len(test_labels))
    
    # Set up callbacks list
    callbacks_list = [checkpoint, early_stopping]
    
    history = train_siamese_network(
        model,
        train_pairs,
        train_labels,
        val_pairs=test_pairs[:max_val_samples],
        val_labels=test_labels[:max_val_samples],
        batch_size=256,  # Increased from 128
        epochs=30,  # Increased from 10
        callbacks=callbacks_list  # Pass callbacks for early stopping and checkpoint
    )
    
    # STEP 7: Plot training history
    plot_training_history(history)
    
    # Load the best model saved during training
    print("\nStep 8: Loading the best saved model...")
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"  Loaded weights from {checkpoint_path}")
    else:
        print("  No saved model found. Using the model from the last epoch.")
    
    # STEP 9: Evaluate the model
    print("\nStep 9: Evaluating the model performance...")
    metrics, predictions, fpr, tpr = evaluate_model(model, test_pairs, test_labels)
    
    # Print the evaluation metrics
    print("\nEvaluation metrics:")
    print("-" * 40)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("-" * 40)
    
    # STEP 10: Plot ROC curve
    print("\nStep 10: Plotting ROC curve...")
    plot_roc_curve(fpr, tpr, metrics['roc_auc'])
    
    # STEP 11: Plot distance distribution
    print("\nStep 11: Plotting distance distribution...")
    plot_distance_distribution(predictions, test_labels)
    
    # STEP 12: Visualize embeddings
    print("\nStep 12: Visualizing embeddings in 2D using t-SNE...")
    # Use all test images or a subset depending on available memory
    max_samples = min(400, len(test_images))
    visualize_embeddings(base_network, test_images[:max_samples], test_labels[:max_samples])
    
    # STEP 13: Visualize some predictions
    print("\nStep 13: Visualizing example predictions...")
    visualize_predictions(model, test_pairs, test_labels, threshold=metrics['optimal_threshold'])
    
    # STEP 14: Perform one-shot learning test
    print("\nStep 14: Performing one-shot learning test...")
    # Print the number of unique classes in the test set
    unique_classes = np.unique(test_labels)
    print(f"  Test dataset contains {len(unique_classes)} unique classes")
    print(f"  Attempting 20-way classification...")
    
    one_shot_accuracy = one_shot_test(model, test_images, test_labels, num_trials=100, num_way=20)
    
    # The one_shot_test function will adjust num_way if needed and print a warning
    print(f"\nOne-shot learning accuracy: {one_shot_accuracy:.4f}")
    print(f"  The model correctly identified the matching character in {one_shot_accuracy*100:.1f}% of trials")
    
    print("\n" + "="*80)
    print(" SIAMESE NETWORK TRAINING COMPLETE ".center(80, "="))
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()