End of Semester Project For Machine Learning For Artificial intelligence Subject - Future University In Egypt (FUE)


Supervised/Unsupervised Machine Learning Models on “Breast Cancer” Dataset Using Ai.

 
# Breast Cancer Diagnosis Prediction

This project aims to predict breast cancer diagnosis using machine learning techniques. It utilizes the Breast Cancer Wisconsin (Diagnostic) dataset, available in the `data.csv` file, containing various features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The target variable, `'diagnosis'`, indicates whether the tumor is malignant (M) or benign (B).

## Workflow Overview

1. **Data Preprocessing:**
   - Load the dataset using Pandas.
   - Separate features (`X`) and the target variable (`y`).
   - Encode the target variable into binary labels (Malignant: 1, Benign: 0).
   - Scale the features using StandardScaler to ensure uniformity in data distribution.

2. **Model Training and Evaluation:**
   - Employ Linear Regression for predicting the diagnosis.
   - Utilize k-fold cross-validation (k=3) to evaluate model performance.
   - Calculate evaluation metrics such as Mean Squared Error (MSE), R-squared, and Mean Absolute Error (MAE) to assess the model's accuracy.

3. **K-means Clustering:**
   - Apply K-means clustering to identify patterns within the dataset.
   - Determine the optimal number of clusters using the Elbow Method.
   - Calculate the Sum of Squared Errors (SSE) and Silhouette Coefficient to evaluate clustering performance.

4. **Visualization:**
   - Plot the clusters obtained from K-means clustering, along with the centroids.

## Instructions

To run the project:

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/username/breast-cancer-diagnosis-prediction.git
   ```

3. Install the required Python libraries using pip:

   ```bash
   pip install pandas scikit-learn matplotlib
   ```

4. Run the `main.py` script:

   ```bash
   python main.py
   ```

5. View the output, including model evaluation metrics and visualizations.

## File Structure

- **main.py:** Entry point of the program. Contains the main workflow, including data preprocessing, model training, and visualization.
  
- **data.csv:** Dataset containing features and target variable.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.
