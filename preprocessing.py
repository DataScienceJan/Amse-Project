import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_path: str) -> pd.DataFrame:
    """
    Loads the AmesHousing dataset from csv_path and cleans it.
    
    Steps include:
      - Dropping columns with >80% missing (but keeps critical columns)
      - Handling missing categorical data
      - Garage & basement imputations
      - Converting 'MS SubClass' to categorical
      - Mapping quality measures (e.g., Kitchen Qual) to numeric
      - Log-transforming SalePrice
      - Final numeric/categorical imputation
      - Outlier capping for selected numeric columns
    """
    try:
        df = pd.read_csv(csv_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}.")
        return pd.DataFrame()
    
    if df.empty:
        print("DataFrame is empty. Check your CSV file.")
        return df

    # 1. Drop columns with >80% missing (but keep certain important ones)
    threshold = 0.8
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
    for must_keep in ['Pool QC', 'GrLivArea', 'Neighborhood']:
        if must_keep in cols_to_drop:
            cols_to_drop.remove(must_keep)

    print("Dropping columns:", cols_to_drop)
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 2. Fill categorical columns that imply 'None'
    for col in ['Alley', 'Fence', 'Misc Feature']:
        if col in df.columns:
            df[col].fillna("None", inplace=True)

    # 3. Pool QC -> 'HasPool' and standardizing categories
    accepted_pool_categories = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
    if 'Pool QC' in df.columns:
        df['HasPool'] = df['Pool QC'].apply(
            lambda x: 1 if pd.notnull(x) and x in accepted_pool_categories else 0
        )
        df['Pool QC'] = df['Pool QC'].apply(
            lambda x: x if pd.notnull(x) and x in accepted_pool_categories else 'NoPool'
        )
    
    # 4. Garage imputations
    if 'Garage Type' in df.columns:
        df['Garage Type'].fillna('NoGarage', inplace=True)
    for gcol in ['Garage Area', 'Garage Cars']:
        if gcol in df.columns and 'Garage Type' in df.columns:
            df.loc[df['Garage Type'] == 'NoGarage', gcol] = 0
    for gcol in ['Garage Cond', 'Garage Finish', 'Garage Qual']:
        if gcol in df.columns:
            df[gcol].fillna('NoGarage', inplace=True)
    if 'Garage Yr Blt' in df.columns:
        df['Garage Yr Blt'].fillna(0, inplace=True)

    # 5. Basement imputations
    bsmt_cols = ['Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2',
                 'Bsmt Qual', 'Bsmt Cond']
    for bcol in bsmt_cols:
        if bcol in df.columns:
            df[bcol].fillna('NoBasement', inplace=True)

    # 6. Masonry veneer
    if 'Mas Vnr Type' in df.columns:
        df['Mas Vnr Type'].fillna('None', inplace=True)
    if 'Mas Vnr Area' in df.columns:
        df['Mas Vnr Area'].fillna(0, inplace=True)

    # 7. Impute 'Lot Frontage' by median in each Neighborhood
    if 'Neighborhood' in df.columns and 'Lot Frontage' in df.columns:
        df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'] \
                               .apply(lambda x: x.fillna(x.median()))

    # 8. Convert 'MS SubClass' to categorical
    if 'MS SubClass' in df.columns:
        df['MS SubClass'] = df['MS SubClass'].astype('category')

    # 9. Fireplace
    if 'Fireplace Qu' in df.columns:
        df['Fireplace Qu'].fillna('NoFireplace', inplace=True)

    # 10. Quality measures -> numeric
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    if 'Kitchen Qual' in df.columns:
        df['Kitchen Qual'] = df['Kitchen Qual'].map(quality_map)
    
    # 11. SalePrice -> log
    if 'SalePrice' in df.columns:
        df['SalePrice_Log'] = np.log(df['SalePrice'])
        plt.figure()
        df['SalePrice_Log'].hist(bins=50)
        plt.title("Histogram of Log-transformed SalePrice")
        plt.xlabel("SalePrice_Log")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # 12. Final numeric/categorical imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    # 13. Additional log transforms for highly skewed numeric features
    # Example set:
    skewed_features = ['Lot Area', '1st Flr SF', 'Mas Vnr Area']
    for feat in skewed_features:
        if feat in df.columns and df[feat].min() > 0:  # avoid log(0)
            new_col = feat + "_Log"
            df[new_col] = np.log1p(df[feat])  # log1p(x) = log(x+1) for stability

    # 14. Outlier capping for multiple numeric columns
    # Example: cap at 99th percentile
    outlier_cols = ['Gr Liv Area', 'Lot Area', 'Mas Vnr Area']
    for oc in outlier_cols:
        if oc in df.columns:
            upper_cap = df[oc].quantile(0.99)
            df.loc[df[oc] > upper_cap, oc] = upper_cap

    print("\nMissing Values After Final Imputation:")
    print(df.isnull().mean()[df.isnull().mean() > 0])
    
    return df
