import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.stats import chi2_contingency
from io import StringIO

# --- Configuration & Caching ---
# @st.cache_resource is used for loading models and resources ONCE
@st.cache_resource
def load_assets():
    """Load the final trained model from the pkl file."""
    try:
        # Load the high-performance LightGBM model
        model = joblib.load('adintel_model.pkl')
        return model
    except FileNotFoundError:
        st.error("MODEL ERROR: 'adintel_model.pkl' not found. Please ensure the file is in the root directory.")
        return None

# --- Data Generation for Demo ---
# This function runs ONLY when the user hasn't uploaded a file.
def generate_sample_data():
    """Creates a sample A/B test dataset with a statistically significant difference."""
    np.random.seed(42) # Ensure the same data is generated every time
    
    # Group A (Control): 10% CTR
    group_a = pd.DataFrame({
        'group': 'A',
        'click': np.random.choice([0, 1], size=15000, p=[0.90, 0.10])
    })
    
    # Group B (Treatment - the clear winner): 13.5% CTR
    group_b = pd.DataFrame({
        'group': 'B',
        'click': np.random.choice([0, 1], size=15000, p=[0.865, 0.135]) 
    })
    
    return pd.concat([group_a, group_b], ignore_index=True)

# --- Statistical Engine ---

def run_chi_squared_test(df_results):
    """Performs the Chi-Squared test on the given DataFrame."""
    # Ensure there are only two groups (A and B)
    groups = df_results['group'].unique()
    if len(groups) != 2:
        return None, None, None

    # Create the contingency table: [Clicks] vs [No Clicks] for Group A vs Group B
    contingency_table = df_results.groupby('group')['click'].value_counts().unstack(fill_value=0)
    
    # Ensure the table has columns for 0 (no click) and 1 (click)
    # This prevents errors if one group had only clicks or only no-clicks
    observed = contingency_table[[0, 1]].values
    
    # Perform the Chi-Squared test
    # The p_value is the crucial result here
    chi2, p_value, dof, expected = chi2_contingency(observed)
    
    return p_value, chi2, observed

def display_verdict(p_value):
    """Displays the final statistical conclusion based on p-value."""
    alpha = 0.05
    st.markdown(f"**P-Value: {p_value:.5f}**")
    
    if p_value < alpha:
        st.success(f"✅ The difference is **STATISTICALLY SIGNIFICANT**. We reject the null hypothesis and conclude Ad B is performing better.")
    else:
        st.warning(f"⚠️ The difference is **NOT STATISTICALLY SIGNIFICANT**. We cannot conclude one ad is better than the other.")

# --- Main Application Logic ---

def main():
    st.set_page_config(page_title="AdIntel: Dashboard", layout="wide")
    st.title("💡 AdIntel: A/B Testing & Performance Dashboard")
    st.markdown("---")

    # Load Model (Optional, for future use)
    # model = load_assets()

    # --- Sidebar for File Upload ---
    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload A/B Test CSV", 
        type=["csv"],
        help="CSV must contain two columns: 'group' (A or B) and 'click' (0 or 1)."
    )

    # --- Load Data ---
    if uploaded_file is not None:
        # User uploaded a file
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        # Use generated data for demo
        df = generate_sample_data()
        st.sidebar.info("Using built-in sample data for demonstration.")

    # --- Data Validation Check ---
    required_cols = ['group', 'click']
    if not all(col in df.columns for col in required_cols):
        st.error(f"❌ Error: CSV must contain columns named 'group' and 'click'.")
        return
    
    # Convert the click column to an integer (0 or 1)
    df['click'] = df['click'].astype(int)

    st.subheader("📊 Campaign Performance Metrics")
    
   # --- Calculate Key Metrics ---
# Call .agg() on the grouped DataFrame, not the column
    metrics = df.groupby('group').agg(
        Impressions=('click', 'count'),
        Clicks=('click', 'sum'),
        CTR=('click', 'mean')
        ).reset_index()

    # --- Display Metrics ---
    col1, col2 = st.columns(2)
    
    # Extract metrics for Group A and B
    metrics_a = metrics[metrics['group'] == 'A'].iloc[0]
    metrics_b = metrics[metrics['group'] == 'B'].iloc[0]

    with col1:
        st.metric(label="Group A (Control) CTR", value=metrics_a['CTR'], delta=f"Impressions: {metrics_a['Impressions']}")
    with col2:
        st.metric(label="Group B (Variant) CTR", value=metrics_b['CTR'], delta=f"Impressions: {metrics_b['Impressions']}")

    # --- Run Statistical Test ---
    st.markdown("---")
    st.subheader("🔬 Statistical Verdict (Chi-Squared Test)")
    
    p_value, chi2, observed = run_chi_squared_test(df)
    
    if p_value is not None:
        display_verdict(p_value)

        # Display the raw contingency table
        st.markdown("##### Observed Click Counts")
        st.table(pd.DataFrame(observed, columns=['No Click (0)', 'Click (1)'], index=['Group A', 'Group B']))
    
    else:
        st.error("Error running test: Data must contain exactly two groups (A and B).")

if __name__ == "__main__":
    main()