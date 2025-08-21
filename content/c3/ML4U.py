"""
Chapter 3: Machine Learning Fundamentals for Power and Utilities
Covers supervised regression/classification, clustering, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, classification_report

def generate_regression_data(samples=200):
    """
    Generate synthetic regression data: temperature vs. daily load (MW).
    """
    temp = np.random.normal(20, 8, samples)  # Ambient temperature (C)
    load = 900 + 15 * temp + np.random.normal(0, 30, samples)  # Linear relationship
    return pd.DataFrame({"Temperature_C": temp, "Load_MW": load})

def regression_example(df):
    """
    Fit and visualize linear regression (Temperature -> Load).
    """
    X = df[["Temperature_C"]]
    y = df["Load_MW"]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="gray", alpha=0.7, label="Observed")
    plt.plot(X, y_pred, color="black", label="Regression Line")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Load (MW)")
    plt.title(f"Linear Regression (MSE = {mse:.2f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapter3_regression.png")
    plt.show()

def generate_classification_data(samples=200):
    """
    Generate synthetic classification data: equipment age, load -> failure probability.
    """
    age = np.random.uniform(1, 30, samples)  # Equipment age (years)
    load = np.random.uniform(500, 1000, samples)  # Transformer load (kVA)
    failure_prob = 1 / (1 + np.exp(-(0.1 * age + 0.002 * load - 7)))  # Sigmoid risk
    failure = np.random.binomial(1, failure_prob)
    return pd.DataFrame({"Age_Years": age, "Load_kVA": load, "Failure": failure})

def classification_example(df):
    """
    Train and evaluate logistic regression for equipment failure prediction.
    """
    X = df[["Age_Years", "Load_kVA"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"]))

def clustering_example():
    """
    Apply clustering to synthetic smart meter load profiles (daily kWh).
    """
    np.random.seed(42)
    cluster1 = np.random.normal(15, 2, (50, 24))  # Low consumption
    cluster2 = np.random.normal(25, 3, (50, 24))  # Medium consumption
    cluster3 = np.random.normal(40, 4, (50, 24))  # High consumption
    data = np.vstack([cluster1, cluster2, cluster3])
    kmeans = KMeans(n_clusters=3, random_state=42).fit(data)

    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.plot(data[kmeans.labels_ == i].T, alpha=0.2)
    plt.xlabel("Hour of Day")
    plt.ylabel("Load (kWh)")
    plt.title("Clustering Smart Meter Daily Load Profiles")
    plt.tight_layout()
    plt.savefig("chapter3_clustering.png")
    plt.show()

if __name__ == "__main__":
    # Regression
    df_reg = generate_regression_data()
    regression_example(df_reg)

    # Classification
    df_class = generate_classification_data()
    classification_example(df_class)

    # Clustering
    clustering_example()
