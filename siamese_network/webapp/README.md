# Enhanced One-Shot Learning Interactive Playground

This web application showcases the power of the Siamese Network model for one-shot learning with an interactive and feature-rich interface. Experience how the model can identify characters it has never seen before with just a single example.

## Overview

The interactive playground demonstrates one-shot learning capabilities by:

1. Showing a random character from the Omniglot dataset as a "target"
2. Displaying a grid of characters (with configurable difficulty levels), including one matching the target
3. Using the trained Siamese Network to predict which character is the match
4. Allowing users to make their own guesses and compare with the AI's predictions
5. Tracking and visualizing performance statistics over time

## How to Run

### Prerequisites

Make sure you have the following installed:
- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`):
  - Flask 2.3+
  - TensorFlow 2.12+
  - NumPy, SciPy
  - PIL (Python Imaging Library)
  - Matplotlib

### Installation

1. Navigate to the webapp directory:
   ```
   cd siamese_network/webapp
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Starting the Application

1. Run the application:
   - On Windows:
     ```
     run.bat
     ```
   - On macOS/Linux:
     ```
     chmod +x run.sh
     ./run.sh
     ```
   - Or directly with Python:
     ```
     python app.py
     ```

2. Open your web browser and go to:
   ```
   http://localhost:5000
   ```

## New Enhanced Features

- **Multiple Difficulty Levels**: Choose between Easy (6 options), Normal (9 options), and Hard (12 options with data augmentation)
- **User Interaction**: Click on characters to make your own guesses and compare with the AI
- **Advanced Statistics**: Track model performance across difficulty levels with detailed statistics
- **Performance Visualization**: View performance trends with an interactive chart
- **Responsive Design**: Improved UI that works well on different screen sizes
- **Data Augmentation**: Hard difficulty includes augmented characters for more challenging tests
- **Tabbed Interface**: Easy navigation between game, statistics, and information
- **User vs. AI Comparison**: See how your guesses compare to the model's predictions

## Technical Details

The enhanced application uses:
- Flask for the backend web server
- Bootstrap 5 with responsive design
- Chart.js for visualizing performance data
- Bootstrap Icons for improved UI
- Customized CSS for a modern look and feel
- Data augmentation techniques for harder challenges
- Session-based statistics tracking

## For Presentations and Demos

This application is perfect for demonstrating your model:

1. Start with Easy difficulty to showcase the basic concept
2. Progress to Normal and Hard to demonstrate model robustness
3. Show the statistics tab to highlight performance across different challenges
4. Let audience members try to beat the AI with their own guesses
5. Explain how Siamese Networks learn similarity rather than classification

## Future Enhancements

Consider these potential future improvements:
- User accounts to track individual performance
- More advanced data augmentation techniques
- Option to visualize the embeddings using dimensionality reduction
- Ability to upload custom characters
- Comparison between different model architectures