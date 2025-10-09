import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def compute_accuracy(y_true, y_pred, threshold=0.5):
    """
    Compute classification accuracy for Siamese Network distance predictions
    
    In a Siamese Network, the output is a distance (not a class prediction).
    We need to convert these distances to binary predictions using a threshold:
    - If distance < threshold: predict as same class (1)
    - If distance >= threshold: predict as different class (0)
    
    Args:
        y_true: True binary labels (1=same class, 0=different class)
        y_pred: Distance predictions from the model (smaller=more similar)
        threshold: Distance threshold for classification
        
    Returns:
        Accuracy: Proportion of correctly classified pairs
    """
    # Convert continuous distances to binary predictions using the threshold
    # For Siamese Networks, smaller distance means more similar:
    # - If distance < threshold: predict as same class (1)
    # - If distance >= threshold: predict as different class (0)
    binary_predictions = np.where(y_pred < threshold, 1, 0)
    
    # Compute accuracy (proportion of correct predictions)
    return accuracy_score(y_true, binary_predictions)

def evaluate_model(model, test_pairs, test_labels):
    """
    Comprehensively evaluate the Siamese Network model performance
    
    This function:
    1. Gets distance predictions from the model for test pairs
    2. Finds the optimal distance threshold using ROC curve analysis
    3. Calculates multiple evaluation metrics for the model
    
    Args:
        model: Trained Siamese Network model
        test_pairs: Test pairs of images [n_pairs, 2, height, width, channels]
        test_labels: Test labels (1=same class, 0=different class)
        
    Returns:
        metrics: Dictionary of evaluation metrics
        predictions: Raw distance predictions
        fpr, tpr: False positive rate and true positive rate for ROC curve
    """
    print("  Getting predictions for test pairs...")
    # Split pairs into left and right images
    test_pairs_left = test_pairs[:, 0]
    test_pairs_right = test_pairs[:, 1]
    
    # Get the distances predicted by the model
    predictions = model.predict([test_pairs_left, test_pairs_right])
    
    # Flatten the predictions (they come as a 2D array with one column)
    predictions = predictions.flatten()
    
    print("  Finding optimal classification threshold...")
    # Find the optimal threshold using ROC curve
    # We use negative predictions because smaller distance = more similar,
    # but ROC curve assumes larger values = more likely positive
    fpr, tpr, thresholds = roc_curve(test_labels, -predictions)
    
    # The optimal threshold maximizes the difference between TPR and FPR
    # (This is one way to find a good threshold - Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"  Optimal distance threshold: {optimal_threshold:.4f}")
    
    # Convert distances to binary predictions using the optimal threshold
    binary_predictions = np.where(predictions < optimal_threshold, 1, 0)
    
    print("  Computing evaluation metrics...")
    # Calculate various performance metrics
    accuracy = accuracy_score(test_labels, binary_predictions)
    precision = precision_score(test_labels, binary_predictions)
    recall = recall_score(test_labels, binary_predictions)
    f1 = f1_score(test_labels, binary_predictions)
    roc_auc = auc(fpr, tpr)
    
    # Create a dictionary with all metrics
    metrics = {
        'accuracy': accuracy,           # Overall accuracy
        'precision': precision,         # Precision (when model predicts "same class")
        'recall': recall,               # Recall (proportion of actual "same class" pairs identified)
        'f1_score': f1,                 # Harmonic mean of precision and recall
        'roc_auc': roc_auc,             # Area under ROC curve (overall discriminative power)
        'optimal_threshold': optimal_threshold  # Best threshold for classification
    }
    
    return metrics, predictions, fpr, tpr

def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot the ROC curve
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under the ROC curve
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def one_shot_test(model, test_images, test_labels, num_trials=20, num_way=20):
    """
    Perform one-shot learning test
    
    Args:
        model: Trained Siamese Network model
        test_images: Test images
        test_labels: Test labels
        num_trials: Number of test trials
        num_way: Number of classes in each trial (N-way)
        
    Returns:
        Accuracy
    """
    n_correct = 0
    classes = np.unique(test_labels)
    
    # Adjust num_way to the number of available classes if needed
    actual_num_way = min(num_way, len(classes))
    if actual_num_way != num_way:
        print(f"Warning: Requested {num_way}-way classification, but only {actual_num_way} classes available.")
        print(f"Proceeding with {actual_num_way}-way classification.")
    
    for _ in range(num_trials):
        # Randomly select support classes
        support_classes = np.random.choice(classes, size=actual_num_way, replace=False)
        
        # Randomly select one query image
        query_class = np.random.choice(support_classes.tolist())
        query_indices = np.where(test_labels == query_class)[0]
        query_idx = np.random.choice(query_indices.tolist())
        query_image = test_images[query_idx]
        
        # Find the nearest neighbor
        min_distance = float('inf')
        predicted_class = None
        
        for support_class in support_classes:
            # Find support image
            support_indices = np.where(test_labels == support_class)[0]
            support_indices = support_indices[support_indices != query_idx]  # Exclude query image
            
            if len(support_indices) > 0:
                support_idx = np.random.choice(support_indices.tolist())
                support_image = test_images[support_idx]
                
                # Calculate distance
                distance = model.predict([
                    query_image.reshape(1, *query_image.shape),
                    support_image.reshape(1, *support_image.shape)
                ])[0][0]
                
                # Update if this is the closest match
                if distance < min_distance:
                    min_distance = distance
                    predicted_class = support_class
        
        # Check if prediction is correct
        if predicted_class == query_class:
            n_correct += 1
    
    # Calculate accuracy
    accuracy = n_correct / num_trials
    return accuracy