import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

# Diagnosis Is The Taget Column
Target_Column = 'diagnosis'

# Separate features and target variable
X = data.drop([Target_Column, 'id'], axis=1)
y = data[Target_Column]

# Encoding the target variable ('M' for malignant = 1 , 'B' for benign = 0)
Target_Label = LabelEncoder()
Y_Axis_Encoded = Target_Label.fit_transform(y)

# Scale the features using StandardScaler
Scaler = StandardScaler()
X_scaled = Scaler.fit_transform(X)

# Creating the Linear Regression model
Model = LinearRegression()

# Applay k=3 for Multiple Linear Regression
Scores = cross_val_score(Model, X_scaled, Y_Axis_Encoded, cv=3)

# Printing the Scores for each fold in Multiple Linear Regression
for i, score in enumerate(Scores):
    print(f"Multiple Linear Regression 'Cross-Validation Fold' {i+1} Score: {score}")

# Calculating the average score for Multiple Linear Regression
Avg_Score = Scores.mean()
print(f"\nMultiple Linear Regression - 'Average Cross-Validation' Score: {Avg_Score}")

Model.fit(X_scaled, Y_Axis_Encoded)

# Predict the target variable
Y_Pred = Model.predict(X_scaled)

# Calculate Mean Squared Error
Mean_Squared_Error = mean_squared_error(Y_Axis_Encoded, Y_Pred)
print(f"\nMean Squared Error: {Mean_Squared_Error}")

# Calculate R_Squared
R_Squared = r2_score(Y_Axis_Encoded, Y_Pred)
print(f"R-squared: {R_Squared}")

# Calculate Mean Absolute Error
Mean_Absolute_Error = mean_absolute_error(Y_Axis_Encoded, Y_Pred)
print(f"Mean Absolute Error: {Mean_Absolute_Error}")

# Determine the optimal number of clusters using the Elbow Method for K-means
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)


optimal_clusters = 20

# Fit KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, random_state=42)
kmeans.fit(X_scaled)

cluster_labels = kmeans.labels_

# Displaying the cluster assignments
data['Cluster'] = cluster_labels
print(data[['id', 'Cluster']])

# Calculate Sum_Squared_Errors
Sum_Squared_Errors = kmeans.inertia_
print(f"\nSum of Squared Errors: {Sum_Squared_Errors}")

# Calculate Silhouette Coefficient
silhouette_coefficient = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette Coefficient: {silhouette_coefficient}")

# Plotting the K-means
plt.figure(figsize=(10, 10))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='o', label='Centroids')
plt.title('K-means Clustering.' ,fontsize=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
