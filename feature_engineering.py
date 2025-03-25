import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

def create_features(df):
    """
    Performs feature engineering on the AmesHousing dataset:
      - Total Bath (Full + 0.5*Half + Basement)
      - Gr Liv Area^2
      - Synthetic DaysOnMarket or transformation
      - Interaction terms: e.g., (Total Bath * Gr Liv Area)
    """

    # 1. Total Bath calculation
    full_bath = df['Full Bath'] if 'Full Bath' in df.columns else 0
    half_bath = df['Half Bath'] if 'Half Bath' in df.columns else 0
    bsmt_full = df['Bsmt Full Bath'] if 'Bsmt Full Bath' in df.columns else 0
    bsmt_half = df['Bsmt Half Bath'] if 'Bsmt Half Bath' in df.columns else 0
    df['Total Bath'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

    # 2. Polynomial feature for Gr Liv Area
    if 'Gr Liv Area' in df.columns:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        gr_liv_area_poly = poly.fit_transform(df[['Gr Liv Area']])
        # Index 1 is the squared term
        df['Gr Liv Area^2'] = gr_liv_area_poly[:, 1]

    # 3. DaysOnMarket: if not present, create synthetic
    if 'DaysOnMarket' not in df.columns:
        df['DaysOnMarket'] = np.random.normal(loc=60, scale=15, size=len(df)).clip(min=1)

    # 4. Possibly add polynomial or log transform for DaysOnMarket
    #    if you suspect a non-linear relationship
    df['DaysOnMarket_Log'] = np.log1p(df['DaysOnMarket'])

    # 5. Create an interaction term: Bath x Gr Liv Area
    if 'Total Bath' in df.columns and 'Gr Liv Area' in df.columns:
        df['Bath_LivArea_Interaction'] = df['Total Bath'] * df['Gr Liv Area']

    # Quick distribution plots for newly created features
    new_features = [
        'Total Bath',
        'Gr Liv Area^2',
        'DaysOnMarket',
        'DaysOnMarket_Log',
        'Bath_LivArea_Interaction'
    ]
    for feature in new_features:
        if feature in df.columns:
            plt.figure()
            df[feature].hist(bins=40)
            plt.title(f"Histogram of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.show()

    return df
# Interview Tip: Be ready to discuss why these features were chosen and how they might improve model performance.