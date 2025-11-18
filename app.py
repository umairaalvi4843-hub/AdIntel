import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.stats import chi2_contingency
from io import StringIO
import os
import time

@st.cache_resource
def load_assets():
    """Load all model assets from disk."""
    assets = {
        'model': 'adintel_model.pkl',
        'maps': 'encoding_maps.pkl',
        'mean': 'global_mean.pkl',
        'features': 'model_features.pkl'
    }
    
    all_files_found = True
    for f in assets.values():
        if not os.path.exists(f):
            st.error(f"Fatal Error: Asset file not found: {f}")
            all_files_found = False

    if not all_files_found:
        st.error("Please download your '.pkl' files from Google Drive (or create them) and place them in the same folder as app.py.")
        return None, None, None, None

    try:
        model = joblib.load(assets['model'])
        encoding_maps = joblib.load(assets['maps'])
        global_mean = joblib.load(assets['mean'])
        model_features = joblib.load(assets['features'])
        
        print("All model assets loaded successfully.")
        return model, encoding_maps, global_mean, model_features
    except Exception as e:
        st.error(f"Error loading model assets: {e}")
        return None, None, None, None

model, encoding_maps, global_mean, model_features = load_assets()

def page_ab_test_analyzer():
    """Renders the A/B Test Analyzer page."""
    st.title("🔬 A/B Test Statistical Analyzer")
    st.markdown("Upload your A/B test results to get an instant, statistically valid conclusion.")
    
    st.sidebar.header("A/B Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload A/B Test CSV", 
        type=["csv"],
        help="CSV must contain two columns: 'group' (A or B) and 'click' (0 or 1)."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return
    else:
        df = generate_sample_data()
        st.sidebar.info("Using built-in sample data for demonstration.")

    required_cols = ['group', 'click']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Error: CSV must contain columns named 'group' and 'click'.")
        return
    
    try:
        df['click'] = pd.to_numeric(df['click'])
    except ValueError:
        st.error("Error: The 'click' column must contain only numbers (0 or 1).")
        return

    st.subheader("📊 Campaign Performance Metrics")
    
    metrics = df.groupby('group').agg(
        Impressions=('click', 'count'),
        Clicks=('click', 'sum'),
        CTR=('click', 'mean')
    ).reset_index()
    
    metrics['CTR_Formatted'] = (metrics['CTR'] * 100).round(3).astype(str) + '%'
    
    col1, col2 = st.columns(2)
    try:
        if 'A' in metrics['group'].values:
            metrics_a = metrics[metrics['group'] == 'A'].iloc[0]
            with col1:
                st.metric(label="Group A (Control) CTR", value=metrics_a['CTR_Formatted'], delta=f"Impressions: {metrics_a['Impressions']}")
        else:
            with col1:
                st.info("Group A not found in data.")

        if 'B' in metrics['group'].values:
            metrics_b = metrics[metrics['group'] == 'B'].iloc[0]
            with col2:
                st.metric(label="Group B (Variant) CTR", value=metrics_b['CTR_Formatted'], delta=f"Impressions: {metrics_b['Impressions']}")
        else:
            with col2:
                st.info("Group B not found in data.")
            
    except IndexError:
        st.error("Error calculating metrics. Please check your 'group' column.")
        return

    st.markdown("---")
    st.subheader("🔬 Statistical Verdict (Chi-Squared Test)")
    
    p_value, chi2, observed = run_chi_squared_test(df)
    
    if p_value is not None and observed is not None:
        display_verdict(p_value)
        st.markdown("##### Observed Click Counts")
        try:
            st.table(pd.DataFrame(observed, columns=['No Click (0)', 'Click (1)'], index=['Group A', 'Group B']))
        except Exception:
            st.warning("Could not display observed click counts table.")
    else:
        st.error("Error running test: Data must contain exactly two groups (e.g., 'A' and 'B').")



def page_live_predictor():
    """Renders the Live CTR Predictor page."""
    st.title("🔮 Live CTR Predictor")
    st.markdown("Input the features of a new ad to get a real-time click probability score from the AI model.")

    if model is None or encoding_maps is None or global_mean is None or model_features is None:
        st.error("Model assets are not loaded. Cannot provide predictions. Check file paths and errors on startup.")
        return

    with st.form(key='prediction_form'):
        st.subheader("Ad Context Features")
        
        try:
            banner_pos_keys = list(encoding_maps['banner_pos'].keys())
            app_cat_keys = list(encoding_maps['app_category'].keys())
            device_type_keys = list(encoding_maps['device_type'].keys())
            conn_type_keys = list(encoding_maps['device_conn_type'].keys())
            site_cat_keys = list(encoding_maps['site_category'].keys())
            c14_keys = list(encoding_maps['C14'].keys())
            c17_keys = list(encoding_maps['C17'].keys())
            c20_keys = list(encoding_maps['C20'].keys())
        except KeyError as e:
            st.error(f"Error: Missing key {e} in 'encoding_maps.pkl'. Please retrain and save your model assets.")
            return

        
        col1, col2 = st.columns(2)
        with col1:
            banner_pos = st.selectbox("Banner Position", options=banner_pos_keys, index=0)
            app_category = st.selectbox("App Category", options=app_cat_keys, index=0)
            device_type = st.selectbox("Device Type", options=device_type_keys, index=0)
        with col2:
            device_conn_type = st.selectbox("Connection Type", options=conn_type_keys, index=0)
            site_category = st.selectbox("Site Category", options=site_cat_keys, index=0)
            hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, 12)

        st.subheader("User Context Features")
        col3, col4 = st.columns(2)
        with col3:
            user_ad_count = st.number_input("How many ads has this user seen today?", min_value=0, value=5)
            device_id = st.text_input("Device ID", value="a99f214a") # A common example ID
        with col4:
            c14 = st.selectbox("C14", options=c14_keys, index=0)
            c17 = st.selectbox("C17", options=c17_keys, index=0)
            c20 = st.selectbox("C20", options=c20_keys, index=0)

        submit_button = st.form_submit_button(label='Predict CTR')

    if submit_button:
        input_data = {
            'banner_pos': banner_pos,
            'site_category': site_category,
            'app_category': app_category,
            'device_type': device_type,
            'device_conn_type': device_conn_type,
            'device_id': device_id,
            'C1': 0, 
            'C14': c14,
            'C15': 0, 
            'C16': 0, 
            'C17': c17,
            'C18': 0, 
            'C19': 0, 
            'C20': c20,
            'C21': 0, 
            'hour_of_day': str(hour_of_day).zfill(2),
            'user_ad_count': user_ad_count,
        }
        
        for col, mapping in encoding_maps.items():
            raw_value = input_data.get(col)
            input_data[f"{col}_encoded"] = mapping.get(raw_value, global_mean)
            
        final_input_df = pd.DataFrame([input_data])
        
        for col in model_features:
            if col not in final_input_df.columns:
                final_input_df[col] = 0 
                
        final_input_df = final_input_df[model_features]
        
       
        for col in final_input_df.columns:
            if col == 'hour_of_day':
                final_input_df[col] = final_input_df[col].astype('category')
            else:
                final_input_df[col] = final_input_df[col].astype(float)
       
        prediction_proba = model.predict_proba(final_input_df)[0, 1] 
        prediction_percent = prediction_proba * 100

        st.subheader("📈 Predicted Click-Through Rate")
        st.metric(label="Click Probability", value=f"{prediction_percent:.2f}%")
        
        with st.expander("Show Raw Model Input (After Encoding & Type Fix)"):
            st.dataframe(final_input_df)
            st.dataframe(final_input_df.dtypes.astype(str), column_config={"0": "Data Type"})

def generate_sample_data():
    """Creates a sample A/B test dataset with a statistically significant difference."""
    np.random.seed(42)
    group_a = pd.DataFrame({'group': 'A', 'click': np.random.choice([0, 1], size=15000, p=[0.90, 0.10])})
    group_b = pd.DataFrame({'group': 'B', 'click': np.random.choice([0, 1], size=15000, p=[0.865, 0.135])})
    return pd.concat([group_a, group_b], ignore_index=True)

def run_chi_squared_test(df_results):
    """Performs the Chi-Squared test on the given DataFrame."""
    groups = df_results['group'].unique()
    if len(groups) != 2 or 'A' not in groups or 'B' not in groups:
        return None, None, None
    
    contingency_table = df_results.groupby('group')['click'].value_counts().unstack(fill_value=0)
    
   
    if 0 not in contingency_table.columns: contingency_table[0] = 0
    if 1 not in contingency_table.columns: contingency_table[1] = 0
        
    if 'A' not in contingency_table.index: contingency_table.loc['A'] = 0
    if 'B' not in contingency_table.index: contingency_table.loc['B'] = 0
        
    observed = contingency_table.loc[['A', 'B'], [0, 1]].values
    
    if observed.sum(axis=1).any() == 0:
        return None, None, None
        
    chi2, p_value, dof, expected = chi2_contingency(observed)
    return p_value, chi2, observed

def display_verdict(p_value):
    """Displays the final statistical conclusion based on p-value."""
    alpha = 0.05
    st.markdown(f"**P-Value: {p_value:.5f}**")
    if p_value < alpha:
        st.success(f"✅ The difference is **STATISTICALLY SIGNIFICANT**.")
    else:
        st.warning(f"⚠️ The difference is **NOT STATISTICALLY SIGNIFICANT**.")

def main():
    st.set_page_config(page_title="AdIntel Dashboard", layout="wide")
    
    st.sidebar.title("AdIntel Navigation")
    page = st.sidebar.radio("Choose a tool", ["A/B Test Analyzer", "Live CTR Predictor"])
    st.sidebar.markdown("---")

    if page == "A/B Test Analyzer":
        page_ab_test_analyzer()
    elif page == "Live CTR Predictor":
        page_live_predictor()

if __name__ == "__main__":
    main()
