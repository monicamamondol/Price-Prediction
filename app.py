import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# --- Setup halaman ---
st.set_page_config(
    page_title="Pharma Price Predictor", 
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #A23B72;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .info-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 2rem 0;
    }
    
    .search-container {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
    }
    
    .about-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
    }
    
    .stTab [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTab [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        border-radius: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
    
    .welcome-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
            
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border: 2px solid #d1d5db;
        border-radius: 10px;
        color: #374151;
        font-size: 14px;
        font-weight: 500;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 5px;
        min-width: 150px;
        padding: 0 15px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
        border-color: #9ca3af;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .stTabs [aria-selected="true"] {
       background-color: #3b82f6 !important;
       border-color: #2563eb !important;
       color: white !important;
       box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .stTabs [data-baseweb="tab-highlight"] {
       background-color: transparent;
    }

    .stTabs [data-baseweb="tab-border"] {
       display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Fungsi Load ---
@st.cache_data
def load_data():
    return pd.read_csv("pharma_data_cleaned.csv")

@st.cache_resource
def load_model():
    return joblib.load("xgb_price_predictor.joblib")

df = load_data()
model = load_model()

# --- Helper Ekstraksi Strength ---
def extract_strength(text):
    if pd.isnull(text): return np.nan
    match = re.search(r"(\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else np.nan

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>💊 Pharma Analytics</h2>
        <p>Drug Price Prediction with AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Dataset Statistics")
    st.metric("Total Products", f"{len(df):,}")
    st.metric("Manufacturers", f"{df['manufacturer'].nunique()}")
    st.metric("Primary Ingredient", f"{df['primary_ingredient'].nunique()}")
    st.metric("Average Price", f"₹{df['price_inr'].mean():.2f}")
    
    st.markdown("### ⚠️ Disclaimer")
    st.info("The application is built for Machine Learning exploration and should not be used as a medical or commercial reference.")

# --- Header utama ---
st.markdown("""
<div class="welcome-banner">
    <h1>💊 Pharma Price Prediction System</h1>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        AI-powered drug price prediction system for the Indian pharmaceutical market
    </p>
</div>
""", unsafe_allow_html=True)

# --- Navigasi Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Price Predictor", "Product Explorer", "About"])

# --- 🏠 HOME ---
with tab1:
    st.markdown('<h2 class="sub-header">Market Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Products</h3>
            <h2>{:,}</h2>
            <p>Active in database</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Manufacturers</h3>
            <h2>{}</h2>
            <p>Different companies</p>
        </div>
        """.format(df['manufacturer'].nunique()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Active Ingredients</h3>
            <h2>{}</h2>
            <p>Unique compounds</p>
        </div>
        """.format(df['primary_ingredient'].nunique()), unsafe_allow_html=True)
    
    with col4:
        avg_price = df['price_inr'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Price</h3>
            <h2>₹{:.0f}</h2>
            <p>Market average</p>
        </div>
        """.format(avg_price), unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Price Distribution by Dosage Form")
        dosage_stats = df.groupby('dosage_form')['price_inr'].agg(['mean', 'count']).reset_index()
        dosage_stats = dosage_stats.sort_values('mean', ascending=False)
        
        fig_bar = px.bar(
            dosage_stats, 
            x='dosage_form', 
            y='mean',
            title='Average Price by Dosage Form',
            color='mean',
            color_continuous_scale='viridis'
        )
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Dosage Form",
            yaxis_title="Average Price (₹)"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### 🏭 Top Manufacturers by Product Count")
        top_manufacturers = df['manufacturer'].value_counts().head(10)
        
        fig_pie = px.pie(
            values=top_manufacturers.values,
            names=top_manufacturers.index,
            title='Market Share by Manufacturer'
        )
        fig_pie.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Data Table
    st.markdown("### Sample Data")
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

# --- 📊 PRICE PREDICTOR ---
with tab2:
    st.markdown('<h2 class="sub-header">AI-Powered Price Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🚀 How it works:</h4>
        <p>Input the drug specifications below and our XGBoost model will predict the market price based on historical data patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("####  Manufacturing Details")
            manufacturer = st.selectbox(
                "Manufacturer",
                sorted(df["manufacturer"].dropna().unique()),
                help="Select the pharmaceutical company"
            )
            dosage_form = st.selectbox(
                "Dosage Form",
                sorted(df["dosage_form"].dropna().unique()),
                help="Select the form of medication (tablet, capsule, etc.)"
            )
            pack_unit = st.selectbox(
                "Pack Unit",
                sorted(df["pack_unit"].dropna().unique()),
                help="Select the packaging unit"
            )
            therapeutic_class = st.selectbox(
                "Therapeutic Class",
                sorted(df["therapeutic_class"].dropna().unique()),
                help="Select the therapeutic category"
            )
            is_discontinued = st.radio(
                "Production Status",
                ["Active", "Discontinued"],
                help="Current production status"
            ) == "Discontinued"
        
        with col2:
            st.markdown("####  Product Specifications")
            pack_size = st.number_input(
                "Pack Size",
                min_value=1.0,
                value=10.0,
                help="Number of units in the pack"
            )
            num_active_ingredients = st.number_input(
                "Number of Active Ingredients",
                min_value=1,
                max_value=2,
                value=1,
                help="Number of active pharmaceutical ingredients"
            )
            primary_ingredient = st.selectbox(
                "Primary Active Ingredient",
                sorted(df["primary_ingredient"].dropna().unique()),
                help="Main active pharmaceutical ingredient"
            )
            primary_strength = st.text_input(
                "Primary Strength (mg)",
                "500",
                help="Strength of the primary ingredient in mg"
            )
        
        submitted = st.form_submit_button("🎯 Predict Price", type="primary")
        
        if submitted:
            input_df = pd.DataFrame([{
                'manufacturer': manufacturer,
                'dosage_form': dosage_form,
                'pack_unit': pack_unit,
                'primary_ingredient': primary_ingredient,
                'therapeutic_class': therapeutic_class,
                'pack_size': pack_size,
                'num_active_ingredients': num_active_ingredients,
                'is_discontinued': int(is_discontinued),
                'primary_strength_mg': extract_strength(primary_strength)
            }])
            
            try:
                pred_log = model.predict(input_df)
                pred_price = np.expm1(pred_log[0])
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h2> Prediction Result</h2>
                    <h1>₹{pred_price:,.2f}</h1>
                    <p>Estimated market price for the specified product</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    similar_products = df[df['primary_ingredient'] == primary_ingredient]
                    avg_similar = similar_products['price_inr'].mean()
                    st.metric(
                        "Similar Products Avg",
                        f"₹{avg_similar:.2f}",
                        f"{((pred_price - avg_similar) / avg_similar * 100):+.1f}%"
                    )
                
                with col2:
                    manufacturer_avg = df[df['manufacturer'] == manufacturer]['price_inr'].mean()
                    st.metric(
                        "Manufacturer Avg",
                        f"₹{manufacturer_avg:.2f}",
                        f"{((pred_price - manufacturer_avg) / manufacturer_avg * 100):+.1f}%"
                    )
                
                with col3:
                    market_avg = df['price_inr'].mean()
                    st.metric(
                        "Market Average",
                        f"₹{market_avg:.2f}",
                        f"{((pred_price - market_avg) / market_avg * 100):+.1f}%"
                    )
                
            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

# --- 🔍 EXPLORER ---
with tab3:
    st.markdown('<h2 class="sub-header">Product Database Explorer</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="search-container">
        <h4>🔎 Advanced Search</h4>
        <p>Search through our comprehensive pharmaceutical database</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        keyword = st.text_input(
            "🔍 Search for active ingredients or brand names",
            placeholder="Enter drug name, ingredient, or brand...",
            help="Search across brand names and active ingredients"
        )
    
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["Both", "Brand Names Only", "Ingredients Only"]
        )
    
    if keyword:
        if search_type == "Both":
            filtered = df[
                df["primary_ingredient"].str.contains(keyword, case=False, na=False) |
                df["brand_name"].str.contains(keyword, case=False, na=False)
            ]
        elif search_type == "Brand Names Only":
            filtered = df[df["brand_name"].str.contains(keyword, case=False, na=False)]
        else:  # Ingredients Only
            filtered = df[df["primary_ingredient"].str.contains(keyword, case=False, na=False)]
        
        st.markdown(f"""
        <div class="info-card">
            <h4>📊 Search Results</h4>
            <p>Found <strong>{len(filtered)}</strong> products matching your search</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(filtered) > 0:
            # Display results
            display_df = filtered[['brand_name', 'manufacturer', 'primary_ingredient', 'therapeutic_class', 'price_inr']].copy()
            display_df['price_inr'] = display_df['price_inr'].apply(lambda x: f"₹{x:,.2f}")
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "brand_name": "Brand Name",
                    "manufacturer": "Manufacturer",
                    "primary_ingredient": "Active Ingredient",
                    "therapeutic_class": "Therapeutic Class",
                    "price_inr": "Price (₹)"
                }
            )
            
            # Quick stats for search results
            if len(filtered) > 1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Price", f"₹{filtered['price_inr'].mean():.2f}")
                
                with col2:
                    st.metric("Price Range", f"₹{filtered['price_inr'].min():.2f} - ₹{filtered['price_inr'].max():.2f}")
                
                with col3:
                    st.metric("Manufacturers", f"{filtered['manufacturer'].nunique()}")
    
    else:
        st.markdown("""
        <div class="info-card">
            <h4>💡 Search Tips</h4>
            <ul>
                <li>Try searching for common drugs like "Paracetamol" or "Aspirin"</li>
                <li>Search by brand names like "Crocin" or "Disprin"</li>
                <li>Use partial matches - "para" will find "Paracetamol"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- 📋 ABOUT ---
with tab4:
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-container">
        <h3>🎯 Project Overview</h3>
        <p><strong>Pharma Price Prediction & Market Analysis</strong> is an AI-powered application that predicts pharmaceutical prices in the Indian market using advanced machine learning techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 Machine Learning Model</h4>
            <ul>
                <li><strong>Algorithm:</strong> XGBoost Regressor</li>
                <li><strong>Target Transform:</strong> Log-transformation</li>
                <li><strong>Features:</strong> 9 key pharmaceutical attributes</li>
                <li><strong>Preprocessing:</strong> OneHotEncoder + StandardScaler</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Model Performance</h4>
            <ul>
                <li><strong>MAE:</strong> ₹38.05</li>
                <li><strong>RMSE:</strong> ₹66.34</li>
                <li><strong>R² Score:</strong> 0.417</li>
                <li><strong>Data:</strong> Outliers removed for better accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>🔧 Technical Features</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div>
                <h5>🎯 Prediction Engine</h5>
                <p>Advanced XGBoost model trained on comprehensive pharmaceutical data</p>
            </div>
            <div>
                <h5>📊 Data Visualization</h5>
                <p>Interactive charts and graphs using Plotly for market insights</p>
            </div>
            <div>
                <h5>🔍 Search & Filter</h5>
                <p>Powerful search capabilities across drug names and ingredients</p>
            </div>
            <div>
                <h5>📱 Responsive Design</h5>
                <p>Modern UI with gradient backgrounds and intuitive navigation</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>Use Cases</h4>
        <ul>
            <li><strong>Pharmaceutical Companies:</strong> Assist in setting competitive prices and understanding the pharmaceutical market landscape.</li>
            <li><strong>Healthcare Providers:</strong> Estimate medication costs for efficient and affordable procurement planning.</li>
            <li><strong>Researchers:</strong> Explore pharmaceutical market trends and develop data-driven competitive insights.</li>
            <li><strong>Regulators:</strong> Monitor market price dynamics and formulate fair, transparent pricing policies.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>© 2025 Pharma Price Prediction System | Monica Mamondol</p>
    <p>Data based on Indian pharmaceutical market • Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%B %Y")), unsafe_allow_html=True)