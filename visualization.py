import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay
import numpy as np
import statsmodels.api as sm
from scipy import stats

def plot_saleprice_distribution(df):
    """
    Plots the distribution of SalePrice and its log-transformation.
    """
    if 'SalePrice' in df.columns:
        plt.figure()
        df['SalePrice'].hist(bins=50)
        plt.xlabel('Sale Price')
        plt.ylabel('Frequency')
        plt.title('Distribution of SalePrice')
        plt.show()

    if 'SalePrice_Log' in df.columns:
        plt.figure()
        df['SalePrice_Log'].hist(bins=50)
        plt.xlabel('Log-transformed SalePrice')
        plt.ylabel('Frequency')
        plt.title('Distribution of Log(SalePrice)')
        plt.show()

def plot_correlation_heatmap(df):
    """
    Plots a heatmap of the correlation matrix for numeric features.
    """
    numeric_features = df.select_dtypes(include=['float64', 'int64'])
    if not numeric_features.empty:
        corr_matrix = numeric_features.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', vmax=1.0, vmin=-1.0, square=True)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

def plot_scatter(df, x, y):
    """
    Creates a scatter plot for two specified features.
    """
    if x in df.columns and y in df.columns:
        plt.figure()
        sns.scatterplot(data=df, x=x, y=y)
        plt.title(f"{x} vs. {y}")
        plt.tight_layout()
        plt.show()

def plot_boxplots_for_features(df, feature_list):
    """
    Creates boxplots for each feature in feature_list.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        feature_list (list of str): List of column names to boxplot.
    """
    for col in feature_list:
        if col in df.columns:
            plt.figure()
            sns.boxplot(y=df[col])
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()

def plot_saleprice_by_neighborhood(df):
    """
    Creates a boxplot of SalePrice by Neighborhood.
    Useful for exploring how prices vary by location.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data. 
                           Must have 'Neighborhood' and 'SalePrice' columns.
    """
    if 'Neighborhood' in df.columns and 'SalePrice' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Neighborhood', y='SalePrice', data=df)
        plt.xticks(rotation=45)
        plt.title('SalePrice by Neighborhood')
        plt.tight_layout()
        plt.show()



# Load Ames Housing dataset
ames = fetch_openml(name="house_prices", as_frame=True, parser='auto')
X = ames.data
y = ames.target

# Select numerical features only for simplicity
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X = X[numerical_features]

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create Partial Dependence Plots
features_to_plot = ['GrLivArea', 'OverallQual', ('GrLivArea', 'OverallQual')]

fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(model, X_train, features=features_to_plot, ax=ax)
plt.tight_layout()
plt.show()



# Generate predictions and residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# QQ Plot
sm.qqplot(residuals, line='45', fit=True, ax=ax1)
ax1.set_title('QQ Plot of Residuals')
ax1.set_xlabel('Theoretical Quantiles')
ax1.set_ylabel('Sample Quantiles')

# Histogram with KDE
sns.histplot(residuals, kde=True, ax=ax2, stat='density')
ax2.set_title('Residual Distribution')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Density')

# Add normal distribution reference
xmin, xmax = ax2.get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
ax2.plot(x, p, 'k', linewidth=2, label='Normal Dist')
ax2.legend()

plt.tight_layout()
plt.show()