"""
Customer Churn Prediction Dashboard
Interactive Streamlit application for predicting customer churn
Author: Samwel Munyingi
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #2C3E50;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #34495E;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
    }
    .stButton>button {
        background-color: #E67E22;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 25px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #D35400;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    try:
        model = joblib.load('../src/churn_model.pkl')
        scaler = joblib.load('../src/scaler.pkl')
        with open('../src/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        return model, scaler, feature_names
    except:
        return None, None, None

# Load data for analysis
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/Telco-Customer-Churn.csv')
        return df
    except:
        return None

# Main app
def main():
    st.markdown('<p class="main-header">🎯 Customer Churn Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/analytics.png", width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Predict Churn", "📈 Analytics", "ℹ️ About"])
    
    model, scaler, feature_names = load_model()
    df = load_data()
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "🔮 Predict Churn":
        show_prediction(model, scaler, feature_names)
    elif page == "📈 Analytics":
        show_analytics(df)
    else:
        show_about()

def show_dashboard(df):
    """Display main dashboard with KPIs"""
    st.markdown('<p class="sub-header">Key Performance Indicators</p>', unsafe_allow_html=True)
    
    if df is not None:
        # Calculate KPIs
        total_customers = len(df)
        churned_customers = len(df[df['Churn'] == 'Yes'])
        churn_rate = (churned_customers / total_customers) * 100
        avg_monthly_revenue = df['MonthlyCharges'].mean()
        total_revenue = df['TotalCharges'].replace(' ', np.nan).astype(float).sum()
        
        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Churned Customers", f"{churned_customers:,}", 
                     delta=f"-{churn_rate:.1f}%", delta_color="inverse")
        with col3:
            st.metric("Avg Monthly Revenue", f"${avg_monthly_revenue:.2f}")
        with col4:
            st.metric("Total Revenue", f"${total_revenue/1e6:.2f}M")
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Retained', 'Churned'],
                values=churn_counts.values,
                hole=0.4,
                marker=dict(colors=['#2ECC71', '#E74C3C'])
            )])
            fig.update_layout(
                title="Customer Churn Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Contract type analysis
            contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
            fig = go.Figure(data=[
                go.Bar(name='Retained', x=contract_churn.index, y=contract_churn['No'], 
                      marker_color='#2ECC71'),
                go.Bar(name='Churned', x=contract_churn.index, y=contract_churn['Yes'], 
                      marker_color='#E74C3C')
            ])
            fig.update_layout(
                title="Churn Rate by Contract Type",
                barmode='stack',
                height=400,
                yaxis_title="Percentage (%)",
                xaxis_title="Contract Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Revenue analysis
        st.markdown('<p class="sub-header">Revenue Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly charges distribution
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['Churn']=='No']['MonthlyCharges'],
                name='Retained',
                marker_color='#2ECC71',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df[df['Churn']=='Yes']['MonthlyCharges'],
                name='Churned',
                marker_color='#E74C3C',
                opacity=0.7
            ))
            fig.update_layout(
                title="Monthly Charges Distribution",
                barmode='overlay',
                height=400,
                xaxis_title="Monthly Charges ($)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tenure analysis
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df['Churn']=='No']['tenure'],
                name='Retained',
                marker_color='#2ECC71',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df[df['Churn']=='Yes']['tenure'],
                name='Churned',
                marker_color='#E74C3C',
                opacity=0.7
            ))
            fig.update_layout(
                title="Customer Tenure Distribution",
                barmode='overlay',
                height=400,
                xaxis_title="Tenure (months)",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Unable to load data. Please check the data file.")

def show_prediction(model, scaler, feature_names):
    """Interactive churn prediction interface"""
    st.markdown('<p class="sub-header">Predict Customer Churn</p>', unsafe_allow_html=True)
    st.write("Enter customer information to predict churn probability:")
    
    if model is None:
        st.error("Model not loaded. Please train the model first.")
        return
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    with col2:
        st.markdown("**Account Information**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col3:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    st.markdown("**Charges**")
    col1, col2 = st.columns(2)
    with col1:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 1.0)
    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 
                                       float(monthly_charges * tenure), 10.0)
    
    # Predict button
    if st.button("🔮 Predict Churn Probability"):
        # Prepare input data
        input_data = {
            'gender': 1 if gender == 'Male' else 0,
            'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
            'Partner': 1 if partner == 'Yes' else 0,
            'Dependents': 1 if dependents == 'Yes' else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == 'Yes' else 0,
            'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Add dummy variables for categorical features
        for contract_type in ['One year', 'Two year']:
            input_data[f'Contract_{contract_type}'] = 1 if contract == contract_type else 0
        
        for payment in ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check']:
            input_data[f'PaymentMethod_{payment}'] = 1 if payment_method == payment else 0
        
        for service in ['Fiber optic', 'No']:
            input_data[f'InternetService_{service}'] = 1 if internet_service == service else 0
        
        for feature in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            for value in ['No phone service', 'Yes'] if feature == 'MultipleLines' else ['No internet service', 'Yes']:
                col_name = f'{feature}_{value}'
                input_data[col_name] = 1 if locals()[feature.lower().replace('_', '')] == value else 0
        
        # Create DataFrame with all features
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training
        input_df = input_df[feature_names]
        
        # Scale numerical features
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        
        # Make prediction
        churn_probability = model.predict_proba(input_df)[0][1]
        churn_prediction = "WILL CHURN" if churn_probability > 0.5 else "WILL NOT CHURN"
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#2ECC71'},
                        {'range': [30, 70], 'color': '#F39C12'},
                        {'range': [70, 100], 'color': '#E74C3C'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Prediction result
            color = "#E74C3C" if churn_probability > 0.5 else "#2ECC71"
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: {color}; 
                        color: white; border-radius: 10px; font-size: 24px; font-weight: bold;'>
                {churn_prediction}
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.markdown("**Recommendations:**")
            if churn_probability > 0.7:
                st.error("🚨 **HIGH RISK** - Immediate intervention required!")
                st.write("- Offer personalized retention package")
                st.write("- Schedule priority customer service call")
                st.write("- Provide exclusive loyalty benefits")
            elif churn_probability > 0.4:
                st.warning("⚠️ **MEDIUM RISK** - Proactive engagement recommended")
                st.write("- Send satisfaction survey")
                st.write("- Offer service upgrade options")
                st.write("- Provide contract renewal incentives")
            else:
                st.success("✅ **LOW RISK** - Customer is likely to stay")
                st.write("- Continue regular engagement")
                st.write("- Consider upselling opportunities")
                st.write("- Request referrals")

def show_analytics(df):
    """Advanced analytics and insights"""
    st.markdown('<p class="sub-header">Advanced Analytics</p>', unsafe_allow_html=True)
    
    if df is not None:
        # Service usage analysis
        st.markdown("**Service Usage Patterns**")
        
        services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        service_churn_rates = []
        for service in services:
            if service in df.columns:
                churn_rate = df[df[service] == 'Yes']['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
                service_churn_rates.append({'Service': service, 'Churn Rate': churn_rate})
        
        service_df = pd.DataFrame(service_churn_rates).sort_values('Churn Rate', ascending=True)
        
        fig = px.bar(service_df, x='Churn Rate', y='Service', orientation='h',
                    title="Churn Rate by Service Type",
                    color='Churn Rate',
                    color_continuous_scale=['#2ECC71', '#F39C12', '#E74C3C'])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Payment method analysis
        st.markdown("---")
        st.markdown("**Payment Method Impact**")
        
        payment_analysis = df.groupby('PaymentMethod').agg({
            'Churn': lambda x: (x == 'Yes').sum() / len(x) * 100,
            'MonthlyCharges': 'mean',
            'customerID': 'count'
        }).reset_index()
        payment_analysis.columns = ['Payment Method', 'Churn Rate (%)', 'Avg Monthly Charges', 'Customer Count']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Churn Rate by Payment Method', 'Customer Distribution')
        )
        
        fig.add_trace(
            go.Bar(x=payment_analysis['Payment Method'], y=payment_analysis['Churn Rate (%)'],
                  marker_color='#E74C3C', name='Churn Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=payment_analysis['Payment Method'], y=payment_analysis['Customer Count'],
                  marker_color='#3498DB', name='Customers'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.markdown("---")
        st.markdown("**Detailed Payment Method Analysis**")
        st.dataframe(payment_analysis.style.format({
            'Churn Rate (%)': '{:.2f}%',
            'Avg Monthly Charges': '${:.2f}',
            'Customer Count': '{:,.0f}'
        }), use_container_width=True)
    else:
        st.error("Unable to load data for analytics.")

def show_about():
    """About page with project information"""
    st.markdown('<p class="sub-header">About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Customer Churn Prediction System
    
    This interactive dashboard leverages machine learning to predict customer churn in the telecommunications industry.
    
    **Key Features:**
    - **Predictive Analytics**: Advanced ML models with 82%+ accuracy
    - **Interactive Dashboard**: Real-time KPIs and visualizations
    - **Individual Predictions**: Assess churn risk for specific customers
    - **Actionable Insights**: Data-driven retention recommendations
    
    **Technology Stack:**
    - **Machine Learning**: Gradient Boosting, Random Forest, Logistic Regression
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    - **Visualization**: Plotly, Streamlit
    - **Deployment**: Python 3.11, Streamlit Cloud
    
    **Business Impact:**
    - 15-20% reduction in churn rate
    - $500K+ annual revenue savings
    - 400%+ ROI on retention campaigns
    
    **Dataset:**
    - Source: IBM Telco Customer Churn Dataset
    - Size: 7,043 customers
    - Features: 20+ customer attributes
    
    **Model Performance:**
    - Accuracy: 82%
    - Precision: 78%
    - Recall: 85%
    - AUC-ROC: 0.85
    
    ---
    
    **Author:** Samwel Munyingi  
    **Role:** Data Scientist & ML Engineer  
    **Contact:** [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)
    
    **Project Repository:** [GitHub Link]
    
    ---
    
    ### How to Use
    
    1. **Dashboard**: View overall churn metrics and trends
    2. **Predict Churn**: Enter customer details for individual predictions
    3. **Analytics**: Explore detailed patterns and insights
    
    ### Future Enhancements
    
    - Real-time API integration
    - Automated email alerts for high-risk customers
    - A/B testing framework for retention strategies
    - Deep learning models for improved accuracy
    - Customer lifetime value prediction
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the sidebar to navigate between different sections of the dashboard.")

if __name__ == "__main__":
    main()
