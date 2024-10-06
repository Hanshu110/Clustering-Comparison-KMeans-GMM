# Clustering Comparison: K-Means vs Gaussian Mixture Model

This project demonstrates a comparison between two popular **unsupervised clustering** algorithms—**K-Means** and **Gaussian Mixture Models (GMM)**. It evaluates the performance of these algorithms on a small dataset, visualizes the results using **Principal Component Analysis (PCA)**, and compares their effectiveness using evaluation metrics like **Inertia**, **Log-Likelihood**, and the **Silhouette Score**.

## Project Structure

- **Dataset**: A custom dataset consisting of 30 samples, each with 3 features representing age, height, and weight.
- **Algorithms**: 
  - K-Means Clustering
  - Gaussian Mixture Model (GMM)
- **Evaluation**: 
  - Inertia (K-Means)
  - Log-Likelihood (GMM)
  - Silhouette Score (both)
- **Visualization**: The clusters formed by each algorithm are visualized in 2D space after dimensionality reduction using **PCA**.

## Clustering Algorithms

### K-Means
K-Means aims to partition the data into `k` clusters by minimizing the sum of squared distances between data points and the nearest cluster centroid. It is a centroid-based clustering method.

- **Inertia**: Measures the compactness of clusters (lower is better).
- **Silhouette Score**: Measures how similar a data point is to its own cluster compared to others.

### Gaussian Mixture Model (GMM)
GMM is a probabilistic model that assumes the data is generated from a mixture of several Gaussian distributions. Each Gaussian represents a cluster, and the algorithm assigns probabilities to each point for belonging to a cluster.

- **Log-Likelihood**: Measures how well the Gaussian distributions fit the data (higher is better).
- **Silhouette Score**: Similar to K-Means, used to evaluate clustering performance.

## Dataset

The dataset contains 30 samples, with 3 features each:

```
[
    [56, 180, 80], [45, 170, 65], [60, 160, 70], [30, 175, 68], [50, 172, 72],
    [40, 165, 75], [55, 150, 60], [65, 140, 85], [48, 190, 77], [52, 155, 62],
    [58, 160, 68], [49, 175, 78], [33, 185, 72], [47, 167, 71], [44, 173, 66],
    [39, 170, 64], [60, 160, 70], [34, 176, 74], [55, 155, 67], [36, 162, 61],
    [50, 178, 76], [61, 159, 65], [40, 150, 72], [51, 145, 63], [62, 172, 70],
    [37, 160, 66], [53, 185, 73], [54, 176, 77], [59, 170, 71], [43, 169, 68]
]
```

### Features
1. Age
2. Height (in cm)
3. Weight (in kg)

## Steps in the Experiment

1. **Data Preprocessing**: The data is standardized using `StandardScaler` to ensure all features have zero mean and unit variance.
   
2. **K-Means Clustering**: The dataset is clustered into 3 clusters using the K-Means algorithm. The performance is evaluated using **Inertia** and **Silhouette Score**.

3. **Gaussian Mixture Model (GMM)**: A GMM with 3 components (clusters) is fitted to the data. The model's performance is evaluated using **Log-Likelihood** and **Silhouette Score**.

4. **Dimensionality Reduction**: The dataset is reduced to 2 dimensions using **Principal Component Analysis (PCA)** to visualize the clusters formed by both algorithms.

5. **Visualization**: Two scatter plots are generated to display the clusters:
    - **K-Means Clustering**
    - **GMM Clustering**

## Evaluation Metrics

- **Inertia (K-Means)**: Lower values indicate tighter clusters.
- **Log-Likelihood (GMM)**: Higher values indicate a better fit of the Gaussian distributions to the data.
- **Silhouette Score**: Ranges from -1 to 1; higher values indicate better-defined clusters.

## Visualizations

The clusters from K-Means and GMM are visualized using PCA for dimensionality reduction. The plots below show how each algorithm groups the data in 2D space:

- **K-Means Clustering**: 
  ![K-Means Visualization]![image](https://github.com/user-attachments/assets/fcc1044f-5440-4f81-8543-d142c84f9653)

  
- **GMM Clustering**: 
  ![GMM Visualization]![image](https://github.com/user-attachments/assets/3821e719-2496-4fbd-85fd-67df0b621610)


## Conclusion

This project demonstrates the differences between K-Means and GMM for clustering. K-Means is faster and more efficient for spherical clusters, while GMM provides a flexible, probabilistic approach that can model more complex cluster shapes. Both methods are evaluated on their cluster quality and visualized using PCA for better understanding.

## Requirements

To run the code, you need the following Python libraries:
- `numpy`
- `scikit-learn`
- `matplotlib`

Install them using:
```bash
pip install numpy scikit-learn matplotlib
```

## How to Run

1. Run the Python script:
   ```bash
   python main.py
   ```

---
