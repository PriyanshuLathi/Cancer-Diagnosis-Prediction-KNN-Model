# Cancer Diagnosis Prediction Model

This project aims to predict whether a tumor is malignant or benign using three different algorithms: K-Nearest Neighbour (KNN), Support Vector Machine (SVM), and Decision Tree. Each model is trained on a dataset containing various tumor characteristics, including radius, texture, perimeter, and area. The primary goal is to classify tumors accurately based on these features and compare the performance of each model.

## Dataset Description

The dataset used in this project classifies breast cancer tumors as malignant (cancerous) or benign (non-cancerous) based on features derived from digitized images of fine needle aspirates (FNA) of breast masses. The dataset contains 569 samples with 33 features, including the target variable. Below is a detailed breakdown of the dataset:

- **Number of Samples:** 569
- **Number of Features:** 33 (including the target variable)
- **Target Variable:** Diagnosis - Indicates the diagnosis of the tumor (M for malignant, B for benign).

## Features

The features in the dataset describe characteristics of the cell nuclei present in the image, grouped into three main types based on their statistical properties:

- **Mean Values:** Average measurements of each feature.
- **Standard Error:** Variability of the measurements.
- **Worst Values:** The largest measurements.

### Key Features:
- **ID:** Unique identifier for each sample.
- **Diagnosis:** Target variable (M = malignant, B = benign).
- **Radius (mean, standard error, worst)**
- **Texture (mean, standard error, worst)**
- **Perimeter (mean, standard error, worst)**
- **Area (mean, standard error, worst)**
- **Smoothness (mean, standard error, worst)**
- **Compactness (mean, standard error, worst)**
- **Concavity (mean, standard error, worst)**
- **Concave Points (mean, standard error, worst)**
- **Symmetry (mean, standard error, worst)**
- **Fractal Dimension (mean, standard error, worst)**

## Handling Missing Values

The dataset was checked for missing values, which were addressed during the preprocessing phase to ensure data quality.

## Data Normalization

Features were normalized to bring them into a similar range, improving the performance and convergence of the models.

## Methodology

The following steps were followed to build and evaluate the models:

1. **Data Loading and Preprocessing:** The dataset was loaded using Pandas. Basic preprocessing steps like handling missing values, encoding categorical variables, and feature scaling were applied where necessary.

2. **Feature Selection:** Relevant features that significantly influence tumor classification were selected using feature importance analysis and correlation heatmaps.

3. **Model Training and Evaluation:**
    - **KNN Model:** The KNN algorithm was used to classify tumors based on their nearest neighbors in the feature space.
    - **SVM Model:** SVM was implemented to find the optimal hyperplane that separates malignant and benign tumors.
    - **Decision Tree Model:** This model used a tree-like structure to split the data into classes based on feature values.

## Model Performance

- **KNN Model:**
    - **Accuracy:** 95%
- **SVM Model:**
    - **Accuracy** 92%
- **Decision Tree Model:**
    - **Accuracy:** 95%

## Further Scope

While the current models achieve high accuracy in predicting tumor malignancy, there are several areas for potential improvement:

- **Feature Engineering:** Additional derived features or transformations could be explored to capture more complex patterns.
- **Model Generalization:** Testing on different datasets or new data can evaluate robustness and generalization.
- **Advanced Models:** Experimenting with algorithms like Random Forests or Neural Networks may improve accuracy.
- **Hyperparameter Optimization:** Techniques like Grid Search or Random Search can fine-tune model performance.
- **Real-World Applications:** Deploying the model as a web app or clinical decision-support tool can provide valuable real-time predictions.

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