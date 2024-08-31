# Cancer Diagnosis Prediction KNN Model

This project aims to predict whether a tumor is malignant or benign using the K-Nearest Neighbour (KNN) algorithm. The model is trained on a dataset containing various tumor characteristics, including radius, texture, perimeter, and area. The primary goal is to classify tumors accurately based on these features.

## Dataset Description

The dataset used in this project is designed to classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous) based on various features derived from digitized images of fine needle aspirates (FNA) of breast masses. The dataset contains 569 samples with 33 features, including the target variable. Below is a detailed breakdown of the dataset:

- Number of Samples: 569
- Number of Features: 33 (including the target variable)
- Target Variable: diagnosis - Indicates the diagnosis of the tumor (M for malignant, B for benign).

## Features

The features in the dataset are computed from a digitized image of a fine needle aspirate of a breast mass. They describe characteristics of the cell nuclei present in the image. The features are grouped into three main types based on their statistical properties:

- Mean Values: Average measurements of each feature.
- Standard Error: Variability of the measurements.
- Worst Values: The largest (mean of the three largest values) measurements.

## Key Features:

- ID: Unique identifier for each sample.
- Diagnosis: Target variable (M = malignant, B = benign).
- Radius (mean, standard error, worst): Mean of distances from the center to points on the perimeter.
- Texture (mean, standard error, worst): Standard deviation of gray-scale values.
- Perimeter (mean, standard error, worst): Perimeter of the tumor.
- Area (mean, standard error, worst): Area of the tumor.
- Smoothness (mean, standard error, worst): Local variation in radius lengths.
- Compactness (mean, standard error, worst): Perimeter^2 / area - 1.0.
- Concavity (mean, standard error, worst): Severity of concave portions of the contour.
- Concave Points (mean, standard error, worst): Number of concave portions of the contour.
- Symmetry (mean, standard error, worst): Symmetry of the tumor shape.
- Fractal Dimension (mean, standard error, worst): Coastline approximation - 1.

## Handling Missing Values

The dataset was checked for missing values, which were addressed during the preprocessing phase to ensure data quality.

## Data Normalization

Features were normalized to bring them into a similar range, improving the performance and convergence of the KNN model.

## Class Distribution

- Malignant (M): Represents cancerous tumors.
- Benign (B): Represents non-cancerous tumors.

## Methodology

The following steps were followed to build and evaluate the model:

1. **Data Loading and Preprocessing**: The dataset was loaded using Pandas. Basic preprocessing steps like handling missing values, encoding categorical variables, and feature scaling were applied where necessary.

2. **Feature Selection**: Relevant features that significantly influence house prices were selected. Feature importance analysis and correlation heatmaps were used to guide the selection.

3. **Model Training**: The KNN model was trained using the Scikit-learn library. The data was split into training and testing sets to ensure unbiased evaluation.

These metrics indicate a near-perfect model performance, with the R² score of 0.9177 demonstrating that the model explains all the variance in the target variable.

## Accuracy and Model Performance

The KNN model achieves an **accuracy of 95%**. This means the model perfectly predicts house prices based on the given features.

## Further Scope

While the current KNN model achieves high accuracy in predicting tumor malignancy, there are several areas for potential improvement and further exploration:

- **Feature Engineering**: Additional derived features or transformations could be explored to capture more complex patterns in the data, enhancing the model's performance.
- **Model Generalization**: Testing the model on different datasets or new data can help evaluate its robustness and ability to generalize to unseen cases.
- **Advanced Models**: Experimenting with more sophisticated algorithms like Support Vector Machines (SVM), Random Forests, or Neural Networks may capture non-linear relationships better and improve accuracy.
- **Hyperparameter Optimization**: Fine-tuning hyperparameters using techniques such as Grid Search or Random Search can further enhance model performance.
- **Real-World Applications**: Deploying the model as a web app or integrating it into a clinical decision-support tool can demonstrate its practicality and provide valuable real-time predictions for healthcare professionals.

## Dependencies

The following libraries are required to run this project:

- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/PriyanshuLathi/Cancer-Diagnosis-Prediction-KNN-Model.git
    ```

2. Install the required Python dependencies:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/PriyanshuLathi/Cancer-Diagnosis-Prediction-KNN-Model/blob/main/LICENSE) file for details.

## Contact

For questions or feedback, feel free to reach out:

- LinkedIn: [Priyanshu Lathi](https://www.linkedin.com/in/priyanshu-lathi)
- GitHub: [Priyanshu Lathi](https://github.com/PriyanshuLathi)

## Authors
- Priyanshu Lathi