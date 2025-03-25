# streamlit_app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import shap  # For feature contribution explanations
import matplotlib.pyplot as plt

# Import your preprocessing and feature engineering functions
from src.preprocessing import load_and_preprocess_data
from src.feature_engineering import create_features

# --------------------------------------------------------------------------------
# 1) HELPER FUNCTIONS
# --------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_training_defaults():
    """
    Load the full AmesHousing CSV, preprocess and feature engineer it,
    then collect default (median or mode) values for every column (except the target).
    Also returns a list of all features and available neighborhoods.
    """
    csv_path = os.path.join("data", "AmesHousing.csv")
    df = load_and_preprocess_data(csv_path)
    if df.empty:
        st.error("Error loading training data for default values.")
        return {}, [], []
    df = create_features(df)
    
    target_col = "SalePrice_Log" if "SalePrice_Log" in df.columns else "SalePrice"
    feature_cols = df.columns.drop(target_col, errors='ignore')
    
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    categorical_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    
    numeric_defaults = df[numeric_cols].median(numeric_only=True).to_dict()
    cat_defaults = df[categorical_cols].mode().iloc[0].to_dict()
    
    full_defaults = {**numeric_defaults, **cat_defaults}
    
    # Neighborhood list
    if "Neighborhood" in df.columns:
        neighborhood_list = sorted(df["Neighborhood"].unique())
    else:
        neighborhood_list = []
    
    # Store min/max for Year Built / Year Remod
    full_defaults["min_year_built"] = int(df["Year Built"].min()) if "Year Built" in df.columns else 1872
    full_defaults["max_year_built"] = int(df["Year Built"].max()) if "Year Built" in df.columns else 2010
    full_defaults["min_year_remod"] = int(df["Year Remod/Add"].min()) if "Year Remod/Add" in df.columns else 1950
    full_defaults["max_year_remod"] = int(df["Year Remod/Add"].max()) if "Year Remod/Add" in df.columns else 2010
    
    return full_defaults, list(feature_cols), neighborhood_list

@st.cache_resource(show_spinner=False)
def load_model(model_choice="Linear (Baseline)"):
    """
    Load the persisted model pipeline (linear, ridge, or random forest).
    """
    model_file_map = {
        "Linear (Baseline)": "linear_model.pkl",
        "Ridge (Tuned)": "ridge_model.pkl",
        "Random Forest": "random_forest.pkl"
    }
    chosen_file = model_file_map.get(model_choice, "linear_model.pkl")
    model_path = os.path.join(os.getcwd(), chosen_file)
    model = joblib.load(model_path)
    return model

# --------------------------------------------------------------------------------
# 2) STREAMLIT PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Ames Housing Price Prediction", layout="wide")
st.title("Ames Housing Price Prediction")

# --------------------------------------------------------------------------------
# 3) PROJECT OVERVIEW
# --------------------------------------------------------------------------------
with st.expander("Project Overview"):
    st.markdown("""
    **Project Pitch:**  
    This tool uses data on property characteristics—like square footage, number of bathrooms, garage size, and neighborhood—to estimate a home’s market value.  
    It highlights which features (e.g., an extra garage stall, finished basement, or pool) add the most value, guiding renovation decisions and price expectations.
    """)

st.markdown("""
This interactive dashboard predicts the sale price of homes in Ames using **multiple models** trained on 40+ features.
Adjust the key features in the sidebar and click **Predict Price**.

**Note:** Features represent key property characteristics. Some features (e.g., Alley, Fence) may be dropped if they have too many missing values. 'Neighborhood' or 'Garage Qual' might become one-hot columns internally.
""")

# --------------------------------------------------------------------------------
# 4) LOAD DEFAULTS & MODEL
# --------------------------------------------------------------------------------
defaults, all_feature_cols, neighborhood_list = load_training_defaults()

st.sidebar.header("Adjust Key Features")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear (Baseline)", "Ridge (Tuned)", "Random Forest"],
    index=0,
    help="Choose which trained model pipeline to use. 'Linear' is simplest, 'Ridge' adds regularization, and 'Random Forest' handles non-linearity."
)

model = load_model(model_choice)

# --------------------------------------------------------------------------------
# 5) SIDEBAR INPUTS
# --------------------------------------------------------------------------------

# ----- Group 1: Basic Info -----
st.sidebar.subheader("Basic Info")
col1, col2 = st.sidebar.columns(2)
with col1:
    if neighborhood_list:
        neighborhood = st.selectbox(
            "Neighborhood",
            neighborhood_list,
            index=0,
            help="Location of the property within Ames."
        )
    else:
        neighborhood = None

with col2:
    default_bath = defaults.get("Total Bath", 2.0)
    total_bath = st.number_input(
        "Total Bathrooms",
        min_value=0.0,
        value=float(default_bath),
        step=0.5,
        help="Full Baths + 0.5 × Half Baths + Basement Baths"
    )

# Year Built / Year Remod
min_year_built = defaults.get("min_year_built", 1872)
max_year_built = defaults.get("max_year_built", 2010)
year_built_default = defaults.get("Year Built", (min_year_built + max_year_built) // 2)
year_built = st.sidebar.slider(
    "Year Built",
    min_value=int(min_year_built),
    max_value=int(max_year_built),
    value=int(year_built_default),
    step=1,
    help="Original construction year."
)

min_year_remod = max(defaults.get("min_year_remod", 1950), year_built)
max_year_remod = defaults.get("max_year_remod", 2010)
year_remod_default = defaults.get("Year Remod/Add", (min_year_remod + max_year_remod) // 2)
year_remod = st.sidebar.slider(
    "Year Remodeled/Added",
    min_value=int(min_year_remod),
    max_value=int(max_year_remod),
    value=int(max(year_remod_default, year_built)),
    step=1,
    help="Year the house was remodeled or had additions (≥ Year Built)."
)

# Days On Market
days_on_market_default = defaults.get("DaysOnMarket", 60.0)
days_on_market = st.sidebar.number_input(
    "Days on Market",
    min_value=1.0,
    value=float(days_on_market_default),
    step=1.0,
    help="Number of days the property has been on the market."
)

# ----- Group 2: Size & Layout -----
st.sidebar.subheader("Size & Layout")

default_gr_liv_area_ft2 = defaults.get("Gr Liv Area", 1500.0)
default_gr_liv_area_m2 = default_gr_liv_area_ft2 / 10.764
living_area_m2 = st.sidebar.number_input(
    "Above-Grade Living Area (m²)",
    min_value=0.0,
    value=float(default_gr_liv_area_m2),
    step=5.0,
    help="Above-grade living area in m². (1 m² ≈ 10.764 ft²)"
)

# ----- Group 3: Garage -----
st.sidebar.subheader("Garage")
garage_cars_default = defaults.get("Garage Cars", 2)
garage_cars = st.sidebar.number_input(
    "Garage Capacity (Cars)",
    min_value=0,
    value=int(garage_cars_default),
    step=1,
    help="Number of cars the garage can accommodate."
)

garage_area_ft2_default = defaults.get("Garage Area", 480.0)
garage_area_m2_default = garage_area_ft2_default / 10.764
garage_area_m2 = st.sidebar.number_input(
    "Garage Area (m²)",
    min_value=0.0,
    value=float(garage_area_m2_default),
    step=5.0,
    help="Garage area in m² (converted to ft²)."
)

# Garage Quality as a selectbox
garage_qual_options = {
    "No Garage": "NoGarage",
    "Excellent (Ex)": "Ex",
    "Good (Gd)": "Gd",
    "Typical (TA)": "TA",
    "Fair (Fa)": "Fa",
    "Poor (Po)": "Po"
}
default_garage_qual = defaults.get("Garage Qual", "NoGarage")
garage_qual_label = [k for k, v in garage_qual_options.items() if v == default_garage_qual]
garage_qual_label = garage_qual_label[0] if garage_qual_label else "No Garage"

selected_garage_qual = st.sidebar.selectbox(
    "Garage Quality",
    list(garage_qual_options.keys()),
    index=list(garage_qual_options.keys()).index(garage_qual_label),
    help="Overall finish/condition of the garage."
)
garage_qual = garage_qual_options[selected_garage_qual]

# ----- Group 4: Basement -----
st.sidebar.subheader("Basement")
basement_qual_options = {
    "No Basement": "NoBasement",
    "Excellent (Ex)": "Ex",
    "Good (Gd)": "Gd",
    "Typical (TA)": "TA",
    "Fair (Fa)": "Fa",
    "Poor (Po)": "Po"
}
default_bsmt_qual = defaults.get("Bsmt Qual", "NoBasement")
bsmt_qual_label = [k for k, v in basement_qual_options.items() if v == default_bsmt_qual]
bsmt_qual_label = bsmt_qual_label[0] if bsmt_qual_label else "No Basement"

selected_bsmt_qual = st.sidebar.selectbox(
    "Basement Quality",
    list(basement_qual_options.keys()),
    index=list(basement_qual_options.keys()).index(bsmt_qual_label),
    help="Overall basement quality."
)
bsmt_qual = basement_qual_options[selected_bsmt_qual]

total_bsmt_ft2_default = defaults.get("Total Bsmt SF", 990.0)
total_bsmt_m2_default = total_bsmt_ft2_default / 10.764
total_bsmt_m2 = st.sidebar.number_input(
    "Total Basement Area (m²)",
    min_value=0.0,
    value=float(total_bsmt_m2_default),
    step=5.0,
    help="Total basement area in m² (converted to ft²)."
)

# ----- Group 5: Kitchen & Extras -----
st.sidebar.subheader("Kitchen & Extras")

# Kitchen Quality
kitchen_qual_options = {
    "Excellent (Ex)": 5,
    "Good (Gd)": 4,
    "Typical (TA)": 3,
    "Fair (Fa)": 2,
    "Poor (Po)": 1
}
kitchen_qual_default_numeric = defaults.get("Kitchen Qual", 3)
# Reverse-map to find the label
rev_map = {v: k for k, v in kitchen_qual_options.items()}
kitchen_qual_label = rev_map.get(kitchen_qual_default_numeric, "Typical (TA)")

selected_kitchen_qual = st.sidebar.selectbox(
    "Kitchen Quality",
    list(kitchen_qual_options.keys()),
    index=list(kitchen_qual_options.keys()).index(kitchen_qual_label),
    help="Kitchen condition from Excellent to Poor."
)
kitchen_qual = kitchen_qual_options[selected_kitchen_qual]

# Pool
has_pool_checkbox = st.sidebar.checkbox(
    "Has Pool?",
    value=bool(defaults.get("HasPool", 0)),
    help="Check if the property has a pool."
)
if has_pool_checkbox:
    pool_quality_options = ["Ex", "Gd", "TA", "Fa", "Po"]
    selected_pool_qual = st.sidebar.selectbox(
        "Pool Quality",
        pool_quality_options,
        index=0,
        help="Overall pool quality (Excellent to Poor)."
    )
    pool_quality = selected_pool_qual
else:
    pool_quality = "NoPool"

# Fireplace
fireplace_options = {
    "No Fireplace": "NoFireplace",
    "Excellent (Ex)": "Ex",
    "Good (Gd)": "Gd",
    "Typical (TA)": "TA",
    "Fair (Fa)": "Fa",
    "Poor (Po)": "Po"
}
default_fireplace = defaults.get("Fireplace Qu", "NoFireplace")
fireplace_label = [k for k, v in fireplace_options.items() if v == default_fireplace]
fireplace_label = fireplace_label[0] if fireplace_label else "No Fireplace"

fireplace_quality = st.sidebar.selectbox(
    "Fireplace Quality",
    list(fireplace_options.keys()),
    index=list(fireplace_options.keys()).index(fireplace_label),
    help="Fireplace condition/quality."
)
fireplace_quality = fireplace_options[fireplace_quality]

# Alley
default_alley = defaults.get("Alley", "None")
alley = st.sidebar.selectbox(
    "Alley Access",
    options=["None", "Grvl", "Pave"],
    index=["None", "Grvl", "Pave"].index(default_alley) if default_alley in ["None", "Grvl", "Pave"] else 0,
    help="Alley access type."
)

# Fence
default_fence = defaults.get("Fence", "None")
fence = st.sidebar.selectbox(
    "Fence Type",
    options=["None", "MnPrv", "GdPrv", "MnWw", "GdWo"],
    index=["None", "MnPrv", "GdPrv", "MnWw", "GdWo"].index(default_fence) if default_fence in ["None", "MnPrv", "GdPrv", "MnWw", "GdWo"] else 0,
    help="Fence type/quality."
)

# --------------------------------------------------------------------------------
# 6) BUILD FINAL FEATURE DICTIONARY
# --------------------------------------------------------------------------------
input_features = dict(defaults)

if neighborhood is not None:
    input_features["Neighborhood"] = neighborhood

# Overwrite defaults with user-chosen values
input_features["Total Bath"] = total_bath
input_features["Year Built"] = year_built
input_features["Year Remod/Add"] = year_remod
input_features["DaysOnMarket"] = days_on_market

# Living Area
gr_liv_area = living_area_m2 * 10.764
input_features["Gr Liv Area"] = gr_liv_area
input_features["Gr Liv Area^2"] = gr_liv_area**2
input_features["Bath_LivArea_Interaction"] = total_bath * gr_liv_area

# Garage
input_features["Garage Cars"] = garage_cars
input_features["Garage Area"] = garage_area_m2 * 10.764
input_features["Garage Qual"] = garage_qual

# Basement
input_features["Bsmt Qual"] = bsmt_qual
input_features["Total Bsmt SF"] = total_bsmt_m2 * 10.764

# Kitchen
input_features["Kitchen Qual"] = kitchen_qual

# Pool & Fireplace
input_features["Pool QC"] = pool_quality
input_features["Fireplace Qu"] = fireplace_quality

# Alley & Fence
input_features["Alley"] = alley
input_features["Fence"] = fence

# Log transform for DaysOnMarket
input_features["DaysOnMarket_Log"] = np.log1p(days_on_market)

# Create a DataFrame
input_df = pd.DataFrame([input_features], columns=all_feature_cols)

# --------------------------------------------------------------------------------
# 7) REVIEW INPUTS (OPTIONAL)
# --------------------------------------------------------------------------------
with st.expander("Review Your Inputs"):
    st.write("Your input settings:")
    st.dataframe(input_df.T)

# --------------------------------------------------------------------------------
# 8) PREDICTION
# --------------------------------------------------------------------------------
if st.button("Predict Price"):
    try:
        prediction_log = model.predict(input_df)
        predicted_price_usd = np.exp(prediction_log[0])  # Convert log(SalePrice) to SalePrice
        
        # "Renovation Bonus" - an optional manual adjustment
        remod_gap = year_remod - year_built
        # Example: 0.2% bump per year gap, capped at 30%
        renovation_bonus_factor = min(1 + remod_gap * 0.002, 1.3)
        adjusted_price = predicted_price_usd * renovation_bonus_factor
        
        st.success(f"### Estimated Sale Price: ${adjusted_price:,.0f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --------------------------------------------------------------------------------
# 9) FEATURE CONTRIBUTIONS (SHAP)
# --------------------------------------------------------------------------------
with st.expander("Feature Contributions"):
    st.write("Approximate dollar contribution for each **key feature** (sorted by absolute impact).")
    try:
        # Attempt to extract regressor and preprocessor from pipeline
        if hasattr(model, "named_steps") and "regressor" in model.named_steps:
            regressor = model.named_steps["regressor"]
            preprocessor = model.named_steps.get("preprocessor", None)
        else:
            regressor = model
            preprocessor = None

        # 1) Partial-match rules => user-friendly names
        partial_feature_map = {
            "Neighborhood":   "Neighborhood",
            "Total Bath":     "Total Bathrooms",
            "Gr Liv Area":    "Living Area",
            "DaysOnMarket":   "Days on Market",
            "Year Built":     "Year Built",
            "Year Remod/Add": "Year Remodeled",
            "Garage Cars":    "Garage Capacity",
            "Garage Area":    "Garage Area",
            "Garage Qual":    "Garage Quality",
            "Kitchen Qual":   "Kitchen Quality",
            "Bsmt Qual":      "Basement Quality",
            "Total Bsmt SF":  "Basement Area",
            "Fireplace Qu":   "Fireplace Quality",
            "Pool QC":        "Pool Quality"
        }

        # 2) Transform input data
        if preprocessor is not None:
            X_processed = preprocessor.transform(input_df)
            try:
                feature_names = preprocessor.get_feature_names_out()
                X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
            except Exception:
                X_processed_df = pd.DataFrame(X_processed)
        else:
            X_processed_df = input_df.copy()

        # 3) Compute SHAP values in log space
        if model_choice == "Random Forest":
            explainer = shap.TreeExplainer(regressor)
            shap_values_all = explainer.shap_values(X_processed_df)
        else:
            explainer = shap.LinearExplainer(regressor, X_processed_df)
            shap_values_all = explainer.shap_values(X_processed_df)

        if isinstance(shap_values_all, list):
            single_shap = shap_values_all[0][0]
        else:
            single_shap = shap_values_all[0]

        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0]

        total_shap_sum = np.sum(single_shap)

        # 4) Sum up SHAP by partial key => approximate dollar effect
        matched_effects = {}
        for partial_key, display_name in partial_feature_map.items():
            matching_cols = [col for col in X_processed_df.columns if partial_key in col]
            if not matching_cols:
                continue
            sum_shap = 0.0
            for mc in matching_cols:
                j = list(X_processed_df.columns).index(mc)
                sum_shap += single_shap[j]
            partial_sum_except_these = base_value + (total_shap_sum - sum_shap)
            effect_j = np.exp(partial_sum_except_these) * (np.exp(sum_shap) - 1.0)
            matched_effects[display_name] = effect_j

        effect_series = pd.Series(matched_effects)
        effect_series = effect_series.reindex(effect_series.abs().sort_values(ascending=False).index)

        if effect_series.empty:
            st.write("No matching key features found in the pipeline. Nothing to display.")
        else:
            # Plot bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["red" if val > 0 else "blue" for val in effect_series]
            effect_series.plot(kind="barh", ax=ax, color=colors)
            ax.set_title("Approx. Dollar Contribution (Key Features Only)")
            ax.set_xlabel("Dollar Effect on Final Price")
            ax.invert_yaxis()
            st.pyplot(fig)

            st.write("**Approx. Dollar Effects (sorted by absolute impact):**")
            st.write(effect_series.apply(lambda x: f"${x:,.0f}"))

    except Exception as e:
        st.write("Error computing SHAP values or dollar effects:", e)

# --------------------------------------------------------------------------------
# 10) HOW PREDICTION WORKS
# --------------------------------------------------------------------------------
with st.expander("How Prediction Works"):
    st.markdown(
        """
1. **Feature Engineering & Encoding**  
   - Numeric features (DaysOnMarket, Total Bath, Gr Liv Area, etc.) may be scaled/log-transformed.
   - Categorical features (Neighborhood, Kitchen Qual, etc.) are one-hot encoded.
2. **Model Equation** (for a log-linear model):
   \[
       \log(SalePrice) = \\beta_0 + \\beta_1 \\times \\text{TotalBath} + \\beta_2 \\times \\text{GrLivArea} + \\dots
   \]
   Then we exponentiate the result to get SalePrice.
3. **Renovation Bonus (Optional)**  
   We add a manual factor for the gap between Year Remod/Add and Year Built. This is purely a demo of how you might layer additional logic.
4. **SHAP for Feature Contributions**  
   We use SHAP values (in log space) to estimate how each feature shifts the prediction in dollar terms.
        """
    )
