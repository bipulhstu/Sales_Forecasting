# 📊 Sales Forecasting Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![Framework](https://img.shields.io/badge/Framework-Streamlit-brightgreen)

A comprehensive sales forecasting solution using Machine Learning and Deep Learning techniques. This project predicts future sales based on historical data, helping businesses with planning, inventory management, and meeting customer demand.

---

## 📝 Project Description

This project develops time series models to forecast future sales based on historical sales data. It experiments with different forecasting techniques including:
- **Statistical Models**: ARIMA, Exponential Smoothing
- **Machine Learning Models**: Random Forest, XGBoost
- **Deep Learning Models**: LSTM (Long Short-Term Memory)

The project includes an interactive **Streamlit Dashboard** for real-time predictions and visualization.

---

## 🎯 Objectives

- Predict future sales based on historical patterns
- Compare performance of different forecasting models
- Provide an interactive web interface for business users
- Help businesses optimize inventory and resource planning

---

## 📂 Project Structure

```
Sales_Forecasting/
│
├── Sales_Forecasting.ipynb         # Part 1: Data Analysis & Initial Models
├── Part_2_Sales_Forecasting.ipynb  # Part 2: Advanced Models & Deployment
├── stores_sales_forecasting.csv    # Dataset
├── app.py                          # Streamlit Dashboard Application
├── README.md                       # Project Documentation
├── requirements.txt                # Python Dependencies
│
├── Models/                         # Saved trained models
│   ├── rf_sales_forecast_model.pkl
│   ├── xgb_sales_forecast_model.pkl
│   ├── lstm_sales_forecast_model.h5
│   └── scaler.pkl
│
└── .ipynb_checkpoints/            # Jupyter checkpoints
```

---

## 🔧 Project Workflow

1. **🧠 Define the Problem**
   - Understand the objective (regression problem)
   - Define success metrics

2. **🗂️ Collect and Prepare Data**
   - Load sales dataset
   - Handle missing values and duplicates
   - Parse dates and aggregate data

3. **📊 Exploratory Data Analysis (EDA)**
   - Visualize sales trends over time
   - Analyze monthly and yearly patterns
   - Identify seasonality and trends

4. **📐 Feature Engineering**
   - Extract date features (Day, Month, Year)
   - Create Day of Week, Week of Year
   - Add weekend indicator
   - Generate lag features

5. **🔀 Split the Data**
   - Time-based train-test split
   - Maintain temporal order

6. **🤖 Choose & Train Models**
   - Random Forest Regressor
   - XGBoost Regressor
   - LSTM Neural Network

7. **📈 Evaluate Models**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - R² Score

8. **🔧 Improve Models**
   - Hyperparameter tuning with GridSearchCV
   - Cross-validation
   - Ensemble methods

9. **🚀 Deploy the Model**
   - Build Streamlit web application
   - Save trained models
   - Create interactive dashboard

---

## 📊 Dataset

**Source:** [Kaggle - Store Sales Forecasting Dataset](https://www.kaggle.com/datasets/tanayatipre/store-sales-forecasting-dataset)

**Features:**
- **Row ID**: Unique identifier
- **Order ID**: Unique order identifier
- **Order Date**: Date when order was placed
- **Ship Date**: Date when order was shipped
- **Ship Mode**: Shipping method
- **Customer ID**: Unique customer identifier
- **Customer Name**: Customer name
- **Segment**: Market segment (Consumer, Corporate, Home Office)
- **Country**: Country
- **City**: City
- **State**: State
- **Postal Code**: Postal code
- **Region**: Geographic region
- **Product ID**: Product identifier
- **Category**: Product category
- **Sub-Category**: Product sub-category
- **Product Name**: Product name
- **Sales**: Total sales amount (Target variable)
- **Quantity**: Quantity purchased
- **Discount**: Discount applied
- **Profit**: Profit generated

**Dataset Statistics:**
- Total Records: 2,121
- Date Range: 2014-2015
- Total Sales: $484,247.50

---

## 🤖 Models & Performance

### 1. Random Forest Regressor (Best Model) ⭐
- **MAE**: 695.89
- **RMSE**: 995.71
- **R² Score**: 0.10
- **Best Parameters**:
  - n_estimators: 100
  - max_depth: 5
  - min_samples_split: 10

### 2. XGBoost Regressor
- **MAE**: 710.28
- **RMSE**: 1,010.74
- **R² Score**: 0.07
- **Best Parameters**:
  - n_estimators: 200
  - learning_rate: 0.01
  - max_depth: 3
  - subsample: 0.7
  - colsample_bytree: 0.7

### 3. LSTM Neural Network
- **MAE**: 898.43
- **RMSE**: 1,365.44
- **R² Score**: -0.70
- **Architecture**:
  - 1 LSTM layer (50 units)
  - Dropout layer (0.2)
  - Dense output layer
  - Optimizer: Adam
  - Loss: MSE

### 4. Statistical Models (Part 1)
- **ARIMA**: Traditional time series forecasting
- **Exponential Smoothing (Holt-Winters)**: Seasonal patterns

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
cd "ML & AI Projects/Sales_Forecasting"
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Additional Dependencies (macOS)
If you're on macOS and encounter XGBoost/LightGBM issues:
```bash
brew install libomp
```

---

## 📦 Dependencies

Main libraries used:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization
- **scikit-learn**: Machine learning models
- **xgboost**: Gradient boosting
- **tensorflow**: Deep learning (LSTM)
- **statsmodels**: Time series analysis
- **streamlit**: Web application framework
- **joblib**: Model serialization

See `requirements.txt` for complete list with versions.

---

## 💻 Usage

### Running Jupyter Notebooks

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open notebooks in order**:
   - First: `Sales_Forecasting.ipynb` (Data analysis & initial models)
   - Second: `Part_2_Sales_Forecasting.ipynb` (Advanced models & deployment)

3. **Run cells sequentially** to:
   - Load and explore data
   - Train models
   - Save trained models

### Running the Streamlit Dashboard

1. **Ensure models are saved** by running cell 22 in `Part_2_Sales_Forecasting.ipynb`

2. **Launch the dashboard**:
```bash
streamlit run app.py
```

3. **Access the dashboard** at `http://localhost:8501`

---

## 🌐 Streamlit Dashboard Features

### 📈 Make Prediction Tab
- Select any date for prediction
- Choose from 3 models or compare all
- Get instant sales forecasts
- View confidence intervals
- Compare with historical averages

### 📊 Historical Data Tab
- View sales statistics
- Interactive time series plots
- Recent sales data table
- Key metrics (Total, Average, Min, Max)

### 📉 Model Performance Tab
- Compare model accuracy
- View error metrics (MAE, RMSE, R²)
- Interactive visualizations
- Model insights and recommendations

### ℹ️ About Tab
- Project overview
- Technologies used
- Model details
- Usage instructions

---

## 📈 Key Insights

1. **Seasonal Patterns**: Sales show strong seasonal trends with peaks during certain months
2. **Weekend Effect**: Weekend sales differ from weekday patterns
3. **Model Performance**: Random Forest provides best accuracy for this dataset
4. **Feature Importance**: Month and Year are strongest predictors
5. **Improvements Needed**: Models show moderate performance, suggesting need for:
   - Additional features (promotions, holidays, competitors)
   - More historical data
   - External factors (economic indicators, weather)

---

## 🔬 Technologies Used

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.8+ |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **ML Frameworks** | Scikit-learn, XGBoost |
| **DL Frameworks** | TensorFlow, Keras |
| **Time Series** | Statsmodels |
| **Web Framework** | Streamlit |
| **Deployment** | Joblib, Pickle |
| **Environment** | Jupyter Notebook, VS Code |

---

## 📊 Model Comparison Visualization

```
Model Performance Comparison:
┌─────────────────────┬──────────┬───────────┬─────────┐
│ Model               │ MAE      │ RMSE      │ R²      │
├─────────────────────┼──────────┼───────────┼─────────┤
│ Random Forest ⭐    │ 695.89   │ 995.71    │ 0.10    │
│ XGBoost             │ 710.28   │ 1,010.74  │ 0.07    │
│ LSTM                │ 898.43   │ 1,365.44  │ -0.70   │
└─────────────────────┴──────────┴───────────┴─────────┘
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Time series forecasting techniques
- ✅ Feature engineering for temporal data
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Model comparison and evaluation
- ✅ Deep learning for sequential data
- ✅ Building interactive ML applications
- ✅ Model deployment with Streamlit
- ✅ End-to-end ML pipeline development

---

## 🔮 Future Improvements

1. **Data Enhancement**
   - Add more historical data (multiple years)
   - Include external factors (holidays, promotions, weather)
   - Incorporate competitor data
   - Add economic indicators

2. **Model Improvements**
   - Try Prophet for time series
   - Implement ensemble methods
   - Add more LSTM architectures
   - Fine-tune hyperparameters further

3. **Feature Engineering**
   - Create lag features
   - Add rolling averages
   - Include product-specific features
   - Geographic features

4. **Dashboard Enhancements**
   - Add multi-step forecasting
   - Include confidence intervals
   - Add model retraining capability
   - Implement A/B testing
   - Add data upload functionality

5. **Deployment**
   - Deploy to cloud (AWS, Azure, GCP)
   - Create Docker container
   - Add CI/CD pipeline
   - Implement API endpoints

---

## 📝 Notes

- **Data Privacy**: This project uses publicly available Kaggle dataset
- **Model Limitations**: Predictions are estimates based on historical patterns
- **Performance**: R² scores indicate room for improvement with additional features
- **Scalability**: Current models handle the dataset efficiently
- **Production Ready**: Dashboard is ready for internal business use

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👨‍💻 Author

**Bipul**
- Project: Sales Forecasting with ML & DL
- Focus: Time Series Analysis, Machine Learning, Deep Learning

---

## 🙏 Acknowledgments

- **Kaggle** for providing the dataset
- **Scikit-learn** community for excellent documentation
- **TensorFlow** team for powerful deep learning framework
- **Streamlit** for amazing web app framework

---

## 📞 Contact & Support

For questions, suggestions, or issues:
- Open an issue in the repository
- Contact the project maintainer

---

## 🌟 Star This Repository

If you find this project helpful, please consider giving it a star! ⭐

---

**Built with ❤️ for Sales Forecasting and Business Intelligence**

*Last Updated: October 2025*


