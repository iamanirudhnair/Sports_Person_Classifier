# Sports Person Classifier Project

## Overview

The **Sports Person Classifier** is a machine learning-based image classification project aimed at recognizing and classifying images of famous sports personalities. This project utilizes the concepts of image processing, wavelet transform, and machine learning techniques such as Support Vector Machines (SVM) for classification.

![image_alt](https://github.com/iamanirudhnair/Real_Estate-Price_Predictor/blob/main/Screenshot%202025-03-09%20085500.png?raw=true)

## Features

- Classifies images of 5 sports personalities: Lionel Messi, Maria Sharapova, Roger Federer, Serena Williams, and Virat Kohli.
- The images are pre-processed using scaling and wavelet transform techniques to extract useful features.
- Uses machine learning algorithms (SVM) for classification and evaluation.
- Implements hyperparameter tuning for model optimization using GridSearchCV.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/Sports_Person_Classifier.git
   cd Sports_Person_Classifier
   ```

2. **Install Dependencies**

   You need to have the following Python libraries installed:

   - `numpy`
   - `opencv-python`
   - `pywt` (PyWavelets)
   - `scikit-learn`
   - `matplotlib` (for visualizing results)
   
   You can install these dependencies using pip:

   ```bash
   pip install numpy opencv-python pywt scikit-learn matplotlib
   ```

## Project Structure

```plaintext
Sports_Person_Classifier/
│
├── model/
│   └── dataset/
│       ├── cropped/
│       │   ├── lionel_messi/
│       │   ├── maria_sharapova/
│       │   ├── roger_federer/
│       │   ├── serena_williams/
│       │   └── virat_kohli/
│
├── Sports_Person_Classifier.py
└── README.md
```

- **dataset/** contains subfolders for each sports personality with their respective images.
- **Sports_Person_Classifier.py** contains the code for image processing, feature extraction, and model training.

## Usage

### Step 1: Prepare the Dataset

The dataset contains images of sports personalities, organized into subdirectories corresponding to each individual. You need to ensure that the dataset is structured as follows:

```plaintext
dataset/
├── cropped/
│   ├── lionel_messi/
│   ├── maria_sharapova/
│   ├── roger_federer/
│   ├── serena_williams/
│   └── virat_kohli/
```

### Step 2: Run the Classifier

To train the model and evaluate the classification accuracy, run the `Sports_Person_Classifier.py` script:

```bash
python Sports_Person_Classifier.py
```

The script will:

1. Load and process images of sports personalities.
2. Perform wavelet transformation on the images.
3. Train a classification model using Support Vector Machines (SVM).
4. Evaluate the model's performance using classification metrics like precision, recall, and F1-score.

### Step 3: Model Evaluation

After running the script, the output will display the performance metrics, including precision, recall, and F1-score for each class (sports personality). An example output might look like:

```
              precision    recall  f1-score   support

           0       0.82      0.75      0.78
           1       0.70      0.90      0.78
           2       1.00      1.00      1.00
           3       1.00      1.00      1.00
           4       0.67      0.80      0.73

    accuracy                           0.70
   macro avg       0.84      0.89      0.86
weighted avg       0.84      0.70      0.74
```

## Key Concepts

### Image Preprocessing

1. **Resizing**: All images are resized to 32x32 pixels to ensure uniformity in size.
2. **Wavelet Transformation**: The Discrete Wavelet Transform (DWT) is applied using the 'db1' wavelet, which helps extract both low and high-frequency components from the image.
3. **Data Scaling**: The pixel values of the images are scaled using `StandardScaler` to ensure proper model training.

### Model Training

- **Support Vector Machine (SVM)**: An SVM model with a radial basis function (RBF) kernel is used for image classification. 
- **Hyperparameter Tuning**: Parameters of the SVM model are optimized using techniques like GridSearchCV.

### Evaluation Metrics

- **Precision**: The proportion of relevant instances among the retrieved instances.
- **Recall**: The proportion of relevant instances that have been retrieved.
- **F1-Score**: The harmonic mean of precision and recall.

## Conclusion

This project demonstrates how to implement an image classification pipeline for recognizing sports personalities using SVM. You can further improve the model by experimenting with different machine learning algorithms, image pre-processing techniques, or by expanding the dataset.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
