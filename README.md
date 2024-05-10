
# Poker-Cards-Recognizing

This repository contains code for a Convolutional Neural Network (CNN) model trained to classify images of playing cards into their respective categories. The model is built using TensorFlow and Keras libraries and utilizes transfer learning with the InceptionV3 architecture.

## Dataset
The dataset used for training, testing, and validation consists of images of playing cards. The dataset is structured into train, test, and validation sets, each containing images belonging to different categories of playing cards.

### Dataset Structure
- **Train Data**: Contains images used for training the model.
- **Test Data**: Contains images used for testing the trained model.
- **Validation Data**: Contains images used for validating the model during training.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook (or Google Colab)


## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/wynalazca382/Poker-Cards-Recognizing
    ```
2. Navigate to the cloned repository:
    ```bash
   cd Poker-Cards-Recognizing
    ```

3. Install the required dependencies:
    ```
    pip install pandas numpy matplotlib tensorflow keras
    ```
4. Download the dataset from Kaggle:
Upload your Kaggle API token (kaggle.json) to the repository.
Run the following commands in the terminal:

    mkdir 1.Dataset
    pip install kaggle
    kaggle datasets download -d gpiosenka/cards-image-datasetclassification
    unzip cards-image-datasetclassification.zip -d 1.Dataset
## Usage

1. Run the Jupyter Notebook (Kartmistrz.ipynb) in your preferred environment (Jupyter Notebook or Google Colab).
2. Follow the instructions and execute the code cells sequentially.
3. Train the model using the provided dataset.
4. Evaluate the model's performance on the test set.
5. Make predictions on new images using the trained model.


## Model Architecture

The CNN model architecture consists of convolutional layers followed by densely connected layers. Here's a summary of the architecture:

1. Input Layer: Convolutional layer with ReLU activation.
2. Max Pooling Layer: Reduces spatial dimensions.
3. Convolutional Layers: Multiple convolutional layers with ReLU activation and max-pooling.
4. Dense Layers: Fully connected layers with ReLU activation for feature extraction.
5. Output Layer: Dense layer with softmax activation for multi-class classification.
## Results and Visualization

After training the model, visualizations of training/validation accuracy and loss are generated to assess model performance. Additionally, the trained model is saved for future use.
## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgments

* The dataset used in this project is provided by Kaggle.
* Inspiration for this project comes from the need for image classification tasks in real-world scenarios.

Please adjust the URLs and file paths accordingly based on your repository structure. Let me know if you need further assistance!