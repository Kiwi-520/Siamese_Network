import numpy as np
import matplotlib.pyplot as plt
from data_utils import create_pairs, get_batch

def train_siamese_network(model, pairs, labels, val_pairs=None, val_labels=None, 
                         batch_size=256, epochs=30, callbacks=None):
    """
    Train the Siamese Network on pairs of images
    
    This function:
    1. Separates the image pairs into left and right images
    2. Sets up training and validation data
    3. Trains the model to distinguish similar from dissimilar pairs
    4. Returns the training history for later analysis
    
    Args:
        model: Compiled Siamese Network model
        pairs: Training pairs [n_pairs, 2, height, width, channels]
        labels: Training labels (1=similar, 0=different) [n_pairs]
        val_pairs: Validation pairs (optional)
        val_labels: Validation labels (optional)
        batch_size: Number of pairs in each batch
        epochs: Number of complete passes through the training data
        
    Returns:
        Training history object containing loss values
    """
    print(f"Step 6: Training the Siamese Network for {epochs} epochs...")
    
    # Separate left and right images from the pairs
    print("  Preparing training data...")
    train_pairs_left = pairs[:, 0]   # All left images 
    train_pairs_right = pairs[:, 1]  # All right images
    
    print(f"  Training with {len(labels)} pairs ({batch_size} pairs per batch)")
    
    # Prepare validation data if provided
    validation_data = None
    if val_pairs is not None and val_labels is not None:
        print(f"  Using {len(val_labels)} pairs for validation")
        val_pairs_left = val_pairs[:, 0]
        val_pairs_right = val_pairs[:, 1]
        validation_data = ([val_pairs_left, val_pairs_right], val_labels)
    
    # Train the model using Keras fit function
    print("  Starting training...")
    history = model.fit(
        [train_pairs_left, train_pairs_right],  # Input: left and right images
        labels,                                # Target: similarity labels
        batch_size=batch_size,                 # Number of pairs per batch
        epochs=epochs,                         # Number of complete passes through the data
        validation_data=validation_data,       # Optional validation data
        callbacks=callbacks,                   # Optional callbacks list 
        verbose=1                              # Show progress bar
    )
    
    print(f"  Training completed! Final loss: {history.history['loss'][-1]:.4f}")
    
    return history

def plot_training_history(history):
    """
    Visualize the training progress by plotting loss curves
    
    This function creates a plot showing how the loss decreased during training,
    which helps assess if:
    - The model is learning (decreasing loss)
    - The model is overfitting (validation loss increases while training loss decreases)
    - Training should continue longer or stop earlier
    
    Args:
        history: Training history object from model.fit()
    """
    print("Step 7: Visualizing training history...")
    
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    
    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    
    # Plot validation loss if it exists
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', 
                 color='orange', linewidth=2, linestyle='--')
        
        # Find the epoch with lowest validation loss
        best_epoch = np.argmin(history.history['val_loss']) 
        best_loss = history.history['val_loss'][best_epoch]
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        plt.text(best_epoch+0.1, best_loss, f'Best epoch: {best_epoch+1}', 
                 fontsize=10, va='center')
    
    plt.title('Siamese Network Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()