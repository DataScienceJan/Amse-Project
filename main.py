import joblib
from src.preprocessing import load_and_preprocess_data
from src.feature_engineering import create_features
from src.visualization import (
    plot_saleprice_distribution,
    plot_correlation_heatmap,
    plot_scatter,
    plot_saleprice_by_neighborhood
)
from src.modeling import train_and_evaluate, train_and_tune, train_and_tune_ensemble

"""
main.py

Orchestrates the AmesHousing data analysis and modeling workflow:
  1. Load & Preprocess Data
  2. Feature Engineering
  3. Visualization
  4. Model Training & Evaluation
  5. Optional: Ensemble Model Tuning
  6. Model Saving
"""

def main():
    # Path to the AmesHousing.csv file
    csv_path = r"C:\Users\47936\OneDrive\Desktop\Prosjekt mappe\ames_project\data\AmesHousing.csv"
    
    # 1. Load and preprocess the data
    df = load_and_preprocess_data(csv_path)
    if df.empty:
        print("No data to process. Exiting.")
        return
    
    # 2. Create additional engineered features
    df = create_features(df)
    
    # 3. Visualize
    plot_saleprice_distribution(df)
    plot_correlation_heatmap(df)
    plot_scatter(df, 'Gr Liv Area', 'SalePrice')
    plot_scatter(df, 'DaysOnMarket', 'SalePrice')
    plot_saleprice_by_neighborhood(df)
    
    # 4a. Basic Linear Regression
    print("\n===== Linear Regression Model =====")
    model_lr = train_and_evaluate(df)
    
    # 4b. Ridge Regression Tuning
    print("\n===== Ridge Regression (Hyperparameter Tuning) =====")
    model_ridge = train_and_tune(df)

    # 5. Ensemble Tuning (Random Forest)
    print("\n===== Random Forest (Hyperparameter Tuning) =====")
    model_rf = train_and_tune_ensemble(df)
    
    # 6. Save the models
    joblib.dump(model_lr, "linear_model.pkl")
    joblib.dump(model_ridge, "ridge_model.pkl")
    joblib.dump(model_rf, "random_forest.pkl")
    
    print("\nModels have been saved: 'linear_model.pkl', 'ridge_model.pkl', 'random_forest.pkl'.")

if __name__ == '__main__':
    main()
