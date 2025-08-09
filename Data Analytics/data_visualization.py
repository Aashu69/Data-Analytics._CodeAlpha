# data_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def plot_histogram(df, column):
    plt.figure(figsize=(8,5))
    sns.histplot(df[column], kde=True, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_boxplot(df, category_col, numerical_col):
    plt.figure(figsize=(8,5))
    sns.boxplot(x=category_col, y=numerical_col, data=df, palette='Set2')
    plt.title(f'Boxplot of {numerical_col} by {category_col}')
    plt.show()

def plot_scatter(df, x_col, y_col, hue_col=None):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col)
    plt.title(f'Scatter Plot of {x_col} vs {y_col}')
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def plot_line(df, date_col, value_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df_sorted = df.sort_values(date_col)

    plt.figure(figsize=(10,6))
    sns.lineplot(x=date_col, y=value_col, data=df_sorted)
    plt.title(f'Time Series of {value_col}')
    plt.xlabel('Date')
    plt.ylabel(value_col)
    plt.show()

def main():
    # Change this to your dataset path
    filepath = 'your_dataset.csv'
    df = load_data(filepath)

    # Customize column names based on your dataset
    numerical_col = 'numerical_column'
    category_col = 'categorical_column'
    x_col = 'num_col1'
    y_col = 'num_col2'
    date_col = 'date_column'

    # Plot visualizations
    plot_histogram(df, numerical_col)
    plot_boxplot(df, category_col, numerical_col)
    plot_scatter(df, x_col, y_col, hue_col=category_col)
    plot_correlation_heatmap(df)
    plot_line(df, date_col, numerical_col)

if __name__ == "__main__":
    main()
