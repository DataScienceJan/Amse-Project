import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def train_and_evaluate(df):
    """
    Splits the dataset into train and test sets, then demonstrates:
      - Cross-validation for performance
      - Final hold-out evaluation
    Uses log-transformed SalePrice if available.
    """
    # 1) Identify target
    target = 'SalePrice_Log' if 'SalePrice_Log' in df.columns else 'SalePrice'

    # 2) Build feature DataFrame
    df_features = df.drop(columns=['SalePrice', 'SalePrice_Log'], errors='ignore')
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    cat_cols = [col for col in df_features.columns if col not in numeric_cols]

    # 3) Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    # 4) Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 5) Cross-validation
    X = df_features
    y = df[target]
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    cv_rmse = -1 * np.mean(cv_scores)
    print("Cross-Validation Results (Linear Regression):")
    print(f"  Mean RMSE (CV=5): {cv_rmse:,.2f}")

    # 6) Hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse_holdout = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_holdout = r2_score(y_test, y_pred)

    print("\nHold-Out Evaluation (Linear Regression):")
    print(f"  Target: {target}")
    print(f"  RMSE: {rmse_holdout:,.2f}")
    print(f"  R2:   {r2_holdout:.4f}")

    return pipeline

def train_and_tune(df):
    """
    GridSearchCV to tune Ridge alpha, plus a final hold-out evaluation.
    """
    target = 'SalePrice_Log' if 'SalePrice_Log' in df.columns else 'SalePrice'
    df_features = df.drop(columns=['SalePrice', 'SalePrice_Log'], errors='ignore')

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    cat_cols = [col for col in df_features.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])

    param_grid = {
        'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 200.0]
    }

    X = df_features
    y = df[target]

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(X, y)

    best_pipeline = grid_search.best_estimator_
    best_cv_rmse = -1 * grid_search.best_score_

    print("\nGridSearchCV Results (Ridge):")
    print("  Best Params:", grid_search.best_params_)
    print(f"  Best CV RMSE: {best_cv_rmse:,.2f}")

    # Final hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    rmse_holdout = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_holdout = r2_score(y_test, y_pred)

    print("\nHold-Out Evaluation (Best Ridge Pipeline):")
    print(f"  Target: {target}")
    print(f"  RMSE: {rmse_holdout:,.2f}")
    print(f"  R2:   {r2_holdout:.4f}")

    return best_pipeline

def train_and_tune_ensemble(df):
    """
    Demonstrates using GridSearchCV to tune a RandomForestRegressor.
    Similar approach can be used for GradientBoostingRegressor, etc.
    """
    target = 'SalePrice_Log' if 'SalePrice_Log' in df.columns else 'SalePrice'
    df_features = df.drop(columns=['SalePrice', 'SalePrice_Log'], errors='ignore')

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    cat_cols = [col for col in df_features.columns if col not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Example parameter grid
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5]
    }

    X = df_features
    y = df[target]

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=3,
        n_jobs=-1
    )
    grid_search.fit(X, y)

    best_pipeline = grid_search.best_estimator_
    best_cv_rmse = -1 * grid_search.best_score_

    print("\nGridSearchCV Results (RandomForestRegressor):")
    print("  Best Params:", grid_search.best_params_)
    print(f"  Best CV RMSE: {best_cv_rmse:,.2f}")

    # Final hold-out
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    rmse_holdout = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_holdout = r2_score(y_test, y_pred)

    print("\nHold-Out Evaluation (Best RandomForestRegressor):")
    print(f"  Target: {target}")
    print(f"  RMSE: {rmse_holdout:,.2f}")
    print(f"  R2:   {r2_holdout:.4f}")

    return best_pipeline
