import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📊 Sales Forecasting Dashboard")
st.markdown("### Predict future sales using Machine Learning & Deep Learning models")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/rf_sales_forecast_model.pkl')
        xgb_model = joblib.load('models/xgb_sales_forecast_model.pkl')
        
        # Try loading LSTM model in order of compatibility
        lstm_model = None
        
        # First try native Keras format (.keras)
        try:
            lstm_model = tf.keras.models.load_model('models/lstm_sales_forecast_model.keras')
            st.success("✅ LSTM model loaded successfully (.keras format)")
        except:
            # Try H5 format with compatibility fixes
            try:
                # Load without compilation to avoid metric issues
                lstm_model = tf.keras.models.load_model('models/lstm_sales_forecast_model.h5', compile=False)
                # Recompile with current TensorFlow version
                lstm_model.compile(optimizer='adam', loss='mse')
                st.success("✅ LSTM model loaded successfully (.h5 format)")
            except Exception as lstm_error:
                st.warning(f"⚠️ Could not load LSTM model: {lstm_error}")
                st.info("💡 LSTM predictions will be unavailable. Random Forest and XGBoost will still work.")
                lstm_model = None
        
        scaler = joblib.load('models/scaler.pkl')
        return rf_model, xgb_model, lstm_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("💡 Try re-running the notebook to save models with current TensorFlow version")
        return None, None, None, None

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('stores_sales_forecasting.csv', encoding='latin1')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        df_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
        df_sales.columns = ['Date', 'Sales']
        df_sales = df_sales.sort_values('Date')
        return df, df_sales
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Feature engineering function
def create_features(date_input, day, month, year, day_of_week, week_of_year, is_weekend):
    features = pd.DataFrame({
        'Day': [day],
        'Month': [month],
        'Year': [year],
        'DayOfWeek': [day_of_week],
        'WeekOfYear': [week_of_year],
        'IsWeekend': [is_weekend]
    })
    return features

# Sidebar - Model Selection
st.sidebar.header("🔧 Configuration")
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Random Forest (Best)", "XGBoost", "LSTM", "All Models Comparison"]
)

# Load models and data
rf_model, xgb_model, lstm_model, scaler = load_models()
df, df_sales = load_data()

if rf_model is None or df is None:
    st.stop()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📈 Make Prediction", "📊 Historical Data", "📉 Model Performance", "ℹ️ About"])

with tab1:
    st.header("Make Sales Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")

        # Date input
        prediction_date = st.date_input(
            "Select Date for Prediction",
            value=datetime.now(),
            min_value=datetime(2014, 1, 1),
            max_value=datetime(2030, 12, 31)
        )

        # Extract features from date
        day = prediction_date.day
        month = prediction_date.month
        year = prediction_date.year
        day_of_week = prediction_date.weekday()
        week_of_year = prediction_date.isocalendar()[1]
        is_weekend = 1 if day_of_week >= 5 else 0

        # Display extracted features
        st.info(f"""
        **Extracted Features:**
        - Day: {day}
        - Month: {month}
        - Year: {year}
        - Day of Week: {day_of_week} ({'Weekend' if is_weekend else 'Weekday'})
        - Week of Year: {week_of_year}
        """)

        # Prediction button
        predict_button = st.button("🔮 Predict Sales", use_container_width=True)

    with col2:
        st.subheader("Prediction Results")

        if predict_button:
            # Create feature dataframe
            features = create_features(prediction_date, day, month, year, day_of_week, week_of_year, is_weekend)

            # Make predictions
            with st.spinner("Making predictions..."):
                if model_choice == "Random Forest (Best)":
                    prediction = rf_model.predict(features)[0]
                    model_name = "Random Forest"
                elif model_choice == "XGBoost":
                    prediction = xgb_model.predict(features)[0]
                    model_name = "XGBoost"
                elif model_choice == "LSTM":
                    if lstm_model is not None:
                        features_scaled = scaler.transform(features)
                        features_reshaped = features_scaled.reshape((1, 1, features_scaled.shape[1]))
                        prediction = lstm_model.predict(features_reshaped, verbose=0)[0][0]
                        model_name = "LSTM"
                    else:
                        st.error("❌ LSTM model is not available. Please select another model.")
                        st.stop()
                else:  # All Models Comparison
                    rf_pred = rf_model.predict(features)[0]
                    xgb_pred = xgb_model.predict(features)[0]
                    
                    # Handle LSTM prediction if model is available
                    if lstm_model is not None:
                        features_scaled = scaler.transform(features)
                        features_reshaped = features_scaled.reshape((1, 1, features_scaled.shape[1]))
                        lstm_pred = lstm_model.predict(features_reshaped, verbose=0)[0][0]
                        
                        st.success("✅ Predictions completed!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Random Forest", f"${rf_pred:,.2f}")
                        with col_b:
                            st.metric("XGBoost", f"${xgb_pred:,.2f}")
                        with col_c:
                            st.metric("LSTM", f"${lstm_pred:,.2f}")
                        
                        # Average prediction with all three models
                        avg_pred = (rf_pred + xgb_pred + lstm_pred) / 3
                        st.markdown(f"### 🎯 Ensemble Average: ${avg_pred:,.2f}")
                        
                        # Visualization with all models
                        fig, ax = plt.subplots(figsize=(10, 6))
                        models = ['Random Forest', 'XGBoost', 'LSTM', 'Ensemble Avg']
                        predictions = [rf_pred, xgb_pred, lstm_pred, avg_pred]
                        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
                    else:
                        st.success("✅ Predictions completed! (LSTM unavailable)")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Random Forest", f"${rf_pred:,.2f}")
                        with col_b:
                            st.metric("XGBoost", f"${xgb_pred:,.2f}")
                        
                        # Average prediction with only two models
                        avg_pred = (rf_pred + xgb_pred) / 2
                        st.markdown(f"### 🎯 Ensemble Average: ${avg_pred:,.2f}")
                        
                        # Visualization with only available models
                        fig, ax = plt.subplots(figsize=(10, 6))
                        models = ['Random Forest', 'XGBoost', 'Ensemble Avg']
                        predictions = [rf_pred, xgb_pred, avg_pred]
                        colors = ['#2ecc71', '#3498db', '#f39c12']

                    bars = ax.bar(models, predictions, color=colors, alpha=0.7, edgecolor='black')
                    ax.set_ylabel('Predicted Sales ($)', fontsize=12, fontweight='bold')
                    ax.set_title(f'Sales Predictions for {prediction_date}', fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)

                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'${height:,.0f}',
                               ha='center', va='bottom', fontweight='bold')

                    st.pyplot(fig)
                    st.stop()

            # Single model prediction
            st.success(f"✅ Prediction completed using {model_name}!")

            # Display prediction
            st.markdown(f"### 🎯 Predicted Sales: ${prediction:,.2f}")

            # Confidence interval (approximation)
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
            st.info(f"**Estimated Range:** ${lower_bound:,.2f} - ${upper_bound:,.2f}")

            # Historical comparison
            if df_sales is not None:
                avg_sales = df_sales['Sales'].mean()
                diff_percent = ((prediction - avg_sales) / avg_sales) * 100

                if diff_percent > 0:
                    st.success(f"📈 {diff_percent:.1f}% above historical average")
                else:
                    st.warning(f"📉 {abs(diff_percent):.1f}% below historical average")

with tab2:
    st.header("📊 Historical Sales Data")

    if df_sales is not None:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Sales", f"${df_sales['Sales'].sum():,.0f}")
        with col2:
            st.metric("Average Daily Sales", f"${df_sales['Sales'].mean():,.0f}")
        with col3:
            st.metric("Max Daily Sales", f"${df_sales['Sales'].max():,.0f}")
        with col4:
            st.metric("Min Daily Sales", f"${df_sales['Sales'].min():,.0f}")

        st.markdown("---")

        # Plot historical sales
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_sales['Date'], df_sales['Sales'], color='steelblue', linewidth=2)
        ax.set_title('Historical Sales Over Time', fontsize=16, fontweight='bold', color='darkmagenta')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f2f2f2')
        st.pyplot(fig)

        # Show data table
        st.subheader("📋 Recent Sales Data")
        st.dataframe(df_sales.tail(20).sort_values('Date', ascending=False), use_container_width=True)

with tab3:
    st.header("📉 Model Performance Comparison")

    # Model metrics (from training)
    metrics_data = {
        'Model': ['Random Forest (Tuned)', 'XGBoost (Tuned)', 'LSTM'],
        'MAE': [695.89, 710.28, 898.43],
        'RMSE': [995.71, 1010.74, 1365.44],
        'R² Score': [0.10, 0.07, -0.70]
    }
    metrics_df = pd.DataFrame(metrics_data)

    st.dataframe(metrics_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                                 .highlight_max(subset=['R² Score'], color='lightgreen'),
                 use_container_width=True)

    st.markdown("---")

    # Visualization
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Accuracy Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics_df['Model']))
        width = 0.25

        ax.bar(x - width, metrics_df['MAE'], width, label='MAE', color='#3498db')
        ax.bar(x, metrics_df['RMSE'], width, label='RMSE', color='#e74c3c')

        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('Error', fontweight='bold')
        ax.set_title('Error Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Model'], rotation=15, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        st.pyplot(fig)

    with col2:
        st.subheader("R² Score Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in metrics_df['R² Score']]
        ax.bar(metrics_df['Model'], metrics_df['R² Score'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Model', fontweight='bold')
        ax.set_ylabel('R² Score', fontweight='bold')
        ax.set_title('Model R² Score', fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15, ha='right')

        st.pyplot(fig)

    st.info("""
    **📌 Model Insights:**
    - **Random Forest** shows the best overall performance with lowest MAE and RMSE
    - **XGBoost** performs similarly to Random Forest
    - **LSTM** requires more data and tuning to achieve better results
    - All models show room for improvement with more features and data
    """)

with tab4:
    st.header("ℹ️ About This Project")

    st.markdown("""
    ## Sales Forecasting Dashboard

    This interactive dashboard uses Machine Learning and Deep Learning models to forecast future sales based on historical data.

    ### 🎯 Project Overview
    - **Objective:** Predict future sales to help businesses with planning, inventory management, and meeting customer demand
    - **Problem Type:** Regression (predicting continuous values)
    - **Dataset:** Store Sales Forecasting Dataset from Kaggle

    ### 🤖 Models Used
    1. **Random Forest Regressor** - Ensemble learning method (Best Performance)
    2. **XGBoost Regressor** - Gradient boosting algorithm
    3. **LSTM** - Deep learning model for time series

    ### 📊 Features
    - Date-based features (Day, Month, Year, Day of Week, Week of Year)
    - Weekend indicator
    - Historical sales trends

    ### 🔧 Technologies
    - **Python** - Programming language
    - **Scikit-learn** - Machine learning models
    - **XGBoost** - Gradient boosting
    - **TensorFlow/Keras** - Deep learning (LSTM)
    - **Streamlit** - Web application framework
    - **Pandas & NumPy** - Data manipulation
    - **Matplotlib & Seaborn** - Visualization

    ### 📈 Model Performance
    - The models were trained on historical sales data
    - Hyperparameter tuning was performed using GridSearchCV
    - Random Forest achieved the best results with:
      - MAE: 695.89
      - RMSE: 995.71
      - R² Score: 0.10

    ### 👨‍💻 Developer
    Built with ❤️ for sales forecasting and business intelligence

    ---

    ### 🚀 How to Use
    1. Navigate to the **Make Prediction** tab
    2. Select a date for prediction
    3. Choose your preferred model
    4. Click **Predict Sales** to see the forecast
    5. Explore historical data and model performance in other tabs

    ### 📝 Note
    Predictions are estimates based on historical patterns. Actual sales may vary due to external factors not captured in the model.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Sales Forecasting Dashboard v1.0 | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
