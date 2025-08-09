import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def load_data(filepath):
    """
    Load dataset from a CSV file.
    """
    df = pd.read_csv(filepath)
    print("Data loaded successfully.")
    return df

def data_overview(df):
    """
    Print basic info and summary statistics.
    """
    print("\n----- Dataset Head -----")
    print(df.head())

    print("\n----- Data Info -----")
    print(df.info())

    print("\n----- Summary Statistics -----")
    print(df.describe())

    print("\n----- Missing Values -----")
    print(df.isnull().sum())

def identify_variable_types(df):
    """
    Identify categorical and numerical variables.
    """
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"\nCategorical variables: {categorical}")
    print(f"Numerical variables: {numerical}")
    return categorical, numerical

def visualize_distributions(df, numerical):
    """
    Plot histograms and boxplots for numerical variables.
    """
    for col in numerical:
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")

        plt.subplot(1,2,2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")

        plt.show()

def correlation_heatmap(df, numerical):
    """
    Plot correlation heatmap for numerical variables.
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df[numerical].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def group_comparison_ttest(df, category_col, numerical_col, group1, group2):
    """
    Perform t-test comparing numerical_col between two groups in category_col.
    """
    group1_data = df[df[category_col] == group1][numerical_col].dropna()
    group2_data = df[df[category_col] == group2][numerical_col].dropna()

    stat, p = ttest_ind(group1_data, group2_data)
    print(f"\nT-test comparing {numerical_col} between {group1} and {group2}:")
    print(f"t-statistic = {stat:.4f}, p-value = {p:.4f}")

def check_missing_data_pattern(df):
    """
    Visualize missing data pattern.
    """
    import missingno as msno
    msno.matrix(df)
    plt.show()

def main():
    # CHANGE this to your dataset path
    filepath = 'your_dataset.csv'

    # Load data
    df = load_data(filepath)

    # Data overview
    data_overview(df)

    # Identify variable types
    categorical, numerical = identify_variable_types(df)

    # Visualize distributions
    visualize_distributions(df, numerical)

    # Correlation heatmap
    correlation_heatmap(df, numerical)

    # Example: if you want to test hypothesis between two groups
    if len(categorical) > 0 and len(numerical) > 0:
        # Replace with your own category and numerical columns & group names
        category_col = categorical[0]
        numerical_col = numerical[0]
        groups = df[category_col].dropna().unique()
        if len(groups) >= 2:
            group_comparison_ttest(df, category_col, numerical_col, groups[0], groups[1])

    # Check missing data pattern (optional)
    try:
        check_missing_data_pattern(df)
    except ImportError:
        print("missingno package not installed; skipping missing data visualization.")

if __name__ == "__main__":
    main()
