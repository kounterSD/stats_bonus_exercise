import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def mean(x):
    return np.average(x)

def weighted_variance(x, w):
    mean = weighted_mean(x, w)
    return np.sum(w * (x - mean) ** 2) / np.sum(w)

def variance(x):
    avg = mean(x)
    return np.average((x-avg)**2)

def weighted_std(x, w):
    return np.sqrt(weighted_variance(x, w))

def std(x):
    return np.sqrt(variance(x))

def weighted_skewness(x, w):
    mean = weighted_mean(x, w)
    std = weighted_std(x, w)
    return np.average((((x - mean) / std) ** 3), weights=w)

def skewness(x):
    n=len(x)
    avg = mean(x)
    sd= std(x)
    skew =np.sum(((x-avg)/sd)**3)
    return (n/((n-1)*(n-2)))*skew

def weighted_covariance(x, y, w):
    mean_x = weighted_mean(x, w)
    mean_y = weighted_mean(y, w)
    return np.sum(w * (x - mean_x) * (y - mean_y)) / np.sum(w)

def weighted_correlation(x, y, w):
    cov = weighted_covariance(x, y, w)
    std_x = weighted_std(x, w)
    std_y = weighted_std(y, w)
    return cov / (std_x * std_y)

def analyze_and_plot_weighted(file_path, city_name):
    df = pd.read_csv(file_path)

    scores = df['SCORE'].values
    temps = df['Avg_Temp'].values
    weights = df['N'].values

    df['Date'] = pd.to_datetime(df['Date'])

    # stats
    mean_score = weighted_mean(scores, weights)
    mean_temp = mean(temps)
    var_score = weighted_variance(scores, weights)
    std_score = weighted_std(scores, weights)
    var_temp = variance(temps)
    std_temp = std(temps)
    skew_score = weighted_skewness(scores, weights)
    skew_temp = skewness(temps)
    corr = weighted_correlation(scores, temps, weights)

    # Weighted regression
    X = temps.reshape(-1, 1)
    y = scores
    reg = LinearRegression()
    reg.fit(X, y, sample_weight=weights)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r_sq = reg.score(X, y, sample_weight=weights)

    # Display
    print(f"\nWeighted Statistical Analysis for {city_name}:\n")
    print(f"Weighted Mean SCORE: {mean_score:.3f}, Mean Temp: {mean_temp:.3f}")
    print(f"Weighted Variance SCORE: {var_score:.3f}, Std Dev SCORE: {std_score:.3f}")
    print(f"Variance Temp: {var_temp:.3f}, Std Dev Temp: {std_temp:.3f}")
    print(f"Weighted Skewness SCORE: {skew_score:.3f}, Skewness Temp: {skew_temp:.3f}")
    print(f"Weighted Correlation between SCORE and Temp: {corr:.3f}")
    print(f"Weighted Regression Equation: SCORE = {slope:.3f} * Temp + {intercept:.3f}")
    print(f"Weighted R-squared: {r_sq:.3f}")

    # Plots
    plt.figure(figsize=(15, 12))

    # Histogram of SCORE (weighted KDE)
    plt.subplot(3, 2, 1)
    sns.histplot(data=df, x='SCORE', weights='N', bins=20, kde=False, color='skyblue')
    plt.title(f'{city_name} - Weighted Distribution of Sentiment Scores')
    plt.xlabel('SCORE')

    # Histogram of Avg_Temp (unweighted because no repeated data)
    plt.subplot(3, 2, 2)
    sns.histplot(temps, bins=20, kde=True, color='salmon')
    plt.title(f'{city_name} - Distribution of Avg Temperature')
    plt.xlabel('Avg_Temp')

    # Boxplot for SCORE (weight-insensitive, use scatter later for clarity)
    plt.subplot(3, 2, 3)
    sns.boxplot(x=scores, color='skyblue')
    plt.title(f'{city_name} - Boxplot of Sentiment Scores')

    # Boxplot for Avg_Temp
    plt.subplot(3, 2, 4)
    sns.boxplot(x=temps, color='salmon')
    plt.title(f'{city_name} - Boxplot of Avg Temperature')

    # Scatterplot with weights as point size
    plt.subplot(3, 2, 5)
    sns.scatterplot(x=temps, y=scores, size=weights, sizes=(10, 200), alpha=0.6)
    x_vals = np.array(plt.gca().get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, color='green', lw=2)
    plt.title(f'{city_name} - SCORE vs Temp (Point size = # Tweets)')
    plt.xlabel('Avg_Temp')
    plt.ylabel('SCORE')

    # SCORE Over Time line plot
    plt.subplot(3,2,6)
    sns.scatterplot(x='Date', y='SCORE', size='N',sizes=(10, 200), alpha=0.5, data=df, marker='o', color='purple')
    plt.title(f'{city_name} - Sentiment Score Over Time')
    plt.xlabel('Date')
    plt.ylabel('SCORE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

file_ny = 'ny_sorted.csv'
file_tx = 'tx_sorted.csv'

analyze_and_plot_weighted(file_ny, 'New York')
analyze_and_plot_weighted(file_tx, 'Texas')
