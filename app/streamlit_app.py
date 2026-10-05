import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models", "session4")

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        margin-bottom: 10px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(128,128,128,0.3);
        text-align: center;
        margin-top: 20px;
    }

    .small-text {
        font-size: 0.85rem;
        opacity: 0.75;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# SESSION STATE
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_model(model_path):
    """
    Load the trained machine learning pipeline/model.
    """
    try:
        return joblib.load(model_path)
    except Exception:
        return None


@st.cache_data
def load_dataset():
    """
    Try to locate and load the telecom churn dataset.
    """

    possible_files = [
        os.path.join(
            DATA_DIR,
            "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        ),
        os.path.join(
            DATA_DIR,
            "telco_customer_churn.csv"
        ),
        os.path.join(
            DATA_DIR,
            "Telco-Customer-Churn.csv"
        ),
        os.path.join(
            DATA_DIR,
            "telco-churn.csv"
        ),
        os.path.join(
            DATA_DIR,
            "cleaned_telco_churn.csv"
        ),
    ]

    for file_path in possible_files:

        if os.path.exists(file_path):

            try:
                return pd.read_csv(file_path)
            except Exception:
                pass

    return None


def find_best_model():
    """
    Look for a saved final/best model.
    """

    preferred_models = [
        "final_model.pkl",
        "best_model.pkl",
        "tuned_model.pkl",
        "random_forest.pkl",
        "logistic_regression.pkl",
        "xgboost.pkl",
        "gradient_boosting.pkl"
    ]

    for model_name in preferred_models:

        model_path = os.path.join(
            MODEL_DIR,
            model_name
        )

        if os.path.exists(model_path):

            return model_path

    if os.path.exists(MODEL_DIR):

        model_files = [
            file
            for file in os.listdir(MODEL_DIR)
            if file.endswith(".pkl")
        ]

        if model_files:

            return os.path.join(
                MODEL_DIR,
                model_files[0]
            )

    return None


def clean_input_data(df):
    """
    Apply the same feature engineering used during model training.
    """

    df = df.copy()

    # --------------------------------------------------------
    # TotalCharges
    # --------------------------------------------------------

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # AvgMonthlyCharge
    # --------------------------------------------------------

    if "AvgMonthlyCharge" not in df.columns:

        df["AvgMonthlyCharge"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"]
        )

    # --------------------------------------------------------
    # TenureGroup
    # --------------------------------------------------------

    if "TenureGroup" not in df.columns:

        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[
                -1,
                12,
                24,
                48,
                60,
                np.inf
            ],
            labels=[
                "0-12",
                "13-24",
                "25-48",
                "49-60",
                "60+"
            ]
        )

    # --------------------------------------------------------
    # NumServices
    # --------------------------------------------------------

    if "NumServices" not in df.columns:

        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        df["NumServices"] = 0

        for col in service_columns:

            if col in df.columns:

                df["NumServices"] += (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("yes")
                    .astype(int)
                )

    return df


def predict_customer(model, customer_df):
    """
    Generate prediction and probability.
    """

    customer_df = clean_input_data(customer_df)

    prediction = model.predict(customer_df)[0]

    probability = None

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(
            customer_df
        )[0]

    return prediction, probability


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📡 Telecom Churn")

    st.markdown("---")

    st.subheader("Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "🔮 Churn Prediction",
            "📊 Dataset",
            "ℹ️ About Project"
        ]
    )

    st.markdown("---")

    st.caption(
        "Machine Learning powered Telecom Customer Churn Prediction"
    )

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

dataset = load_dataset()

model_path = find_best_model()

model = None

if model_path:

    model = load_model(model_path)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("📡 Telecom Customer Churn Dashboard")

    st.markdown(
        """
        ### Customer Retention & Churn Analytics

        Analyze customer behavior, identify churn patterns,
        and understand the key business factors associated
        with customer attrition.
        """
    )

    st.markdown("---")

    # ========================================================
    # CHECK DATASET
    # ========================================================

    if dataset is None:

        st.error(
            "❌ Telecom dataset could not be found."
        )

        st.info(
            f"""
            Please make sure your CSV file is inside:

            `{DATA_DIR}`

            Supported filenames:

            - WA_Fn-UseC_-Telco-Customer-Churn.csv
            - telco_customer_churn.csv
            - Telco-Customer-Churn.csv
            - telco-churn.csv
            - cleaned_telco_churn.csv
            """
        )

    else:

        # ====================================================
        # PREPARE DASHBOARD DATA
        # ====================================================

        dashboard_df = dataset.copy()

        # ----------------------------------------------------
        # Clean Churn column
        # ----------------------------------------------------

        if "Churn" in dashboard_df.columns:

            dashboard_df["Churn_Clean"] = (
                dashboard_df["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            dashboard_df["Churn_Label"] = (
                dashboard_df["Churn_Clean"]
                .map({
                    "yes": "Churned",
                    "1": "Churned",
                    "true": "Churned",
                    "churn": "Churned",
                    "churned": "Churned",
                    "no": "Stayed",
                    "0": "Stayed",
                    "false": "Stayed",
                    "stay": "Stayed",
                    "stayed": "Stayed"
                })
            )

            # Handle unexpected values safely
            dashboard_df["Churn_Label"] = (
                dashboard_df["Churn_Label"]
                .fillna(
                    dashboard_df["Churn"]
                    .astype(str)
                    .str.strip()
                )
            )

        # ----------------------------------------------------
        # Numeric columns
        # ----------------------------------------------------

        if "MonthlyCharges" in dashboard_df.columns:

            dashboard_df["MonthlyCharges"] = pd.to_numeric(
                dashboard_df["MonthlyCharges"],
                errors="coerce"
            )

        if "TotalCharges" in dashboard_df.columns:

            dashboard_df["TotalCharges"] = pd.to_numeric(
                dashboard_df["TotalCharges"],
                errors="coerce"
            )

        if "tenure" in dashboard_df.columns:

            dashboard_df["tenure"] = pd.to_numeric(
                dashboard_df["tenure"],
                errors="coerce"
            )

        # ====================================================
        # KPI CALCULATIONS
        # ====================================================

        total_customers = len(dashboard_df)

        # ----------------------------------------------------
        # Churn KPIs
        # ----------------------------------------------------

        if "Churn_Label" in dashboard_df.columns:

            churned = (
                dashboard_df["Churn_Label"]
                .eq("Churned")
                .sum()
            )

            stayed = (
                dashboard_df["Churn_Label"]
                .eq("Stayed")
                .sum()
            )

            churn_rate = (
                churned / total_customers * 100
                if total_customers > 0
                else 0
            )

        else:

            churned = 0
            stayed = 0
            churn_rate = 0

        # ----------------------------------------------------
        # Average Monthly Charges
        # ----------------------------------------------------

        if "MonthlyCharges" in dashboard_df.columns:

            avg_monthly_charges = (
                dashboard_df["MonthlyCharges"]
                .mean()
            )

        else:

            avg_monthly_charges = 0

        # ----------------------------------------------------
        # Average Tenure
        # ----------------------------------------------------

        if "tenure" in dashboard_df.columns:

            avg_tenure = (
                dashboard_df["tenure"]
                .mean()
            )

        else:

            avg_tenure = 0

        # ====================================================
        # FIVE KPIs
        # ====================================================

        st.subheader("📌 Key Performance Indicators")

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

        with kpi1:

            st.metric(
                label="👥 Total Customers",
                value=f"{total_customers:,}"
            )

        with kpi2:

            st.metric(
                label="⚠️ Churned Customers",
                value=f"{churned:,}"
            )

        with kpi3:

            st.metric(
                label="📉 Churn Rate",
                value=f"{churn_rate:.2f}%"
            )

        with kpi4:

            st.metric(
                label="💰 Avg Monthly Charges",
                value=f"${avg_monthly_charges:,.2f}"
            )

        with kpi5:

            st.metric(
                label="⏳ Avg Tenure",
                value=f"{avg_tenure:.1f} months"
            )

        st.markdown("---")

        # ====================================================
        # IMPORT PLOTLY
        # ====================================================

        import plotly.express as px
        import plotly.graph_objects as go

        # ====================================================
        # CHART 1
        # CUSTOMER CHURN DISTRIBUTION
        # ====================================================

        st.subheader("📊 1. Customer Churn Distribution")

        if "Churn_Label" in dashboard_df.columns:

            churn_distribution = (
                dashboard_df["Churn_Label"]
                .value_counts()
                .rename_axis("Customer Status")
                .reset_index(name="Customers")
            )

            fig1 = px.pie(
                churn_distribution,
                names="Customer Status",
                values="Customers",
                hole=0.45,
                title="Customers by Churn Status"
            )

            fig1.update_traces(
                textposition="inside",
                textinfo="percent+label"
            )

            fig1.update_layout(
                height=430,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                ),
                legend_title_text="Status"
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            st.caption(
                f"Overall churn rate: {churn_rate:.2f}% "
                f"({churned:,} of {total_customers:,} customers)."
            )

        else:

            st.warning(
                "Churn column is not available in the dataset."
            )

        st.markdown("---")

        # ====================================================
        # CHART 2
        # CHURN BY CONTRACT TYPE
        # ====================================================

        st.subheader("📄 2. Churn by Contract Type")

        if (
            "Contract" in dashboard_df.columns
            and "Churn_Label" in dashboard_df.columns
        ):

            contract_data = (
                dashboard_df[
                    dashboard_df["Churn_Label"]
                    .isin(["Churned", "Stayed"])
                ]
                .groupby(
                    ["Contract", "Churn_Label"]
                )
                .size()
                .reset_index(name="Customers")
            )

            contract_order = [
                "Month-to-month",
                "One year",
                "Two year"
            ]

            contract_data["Contract"] = pd.Categorical(
                contract_data["Contract"],
                categories=contract_order,
                ordered=True
            )

            contract_data = contract_data.sort_values(
                "Contract"
            )

            fig2 = px.bar(
                contract_data,
                x="Contract",
                y="Customers",
                color="Churn_Label",
                barmode="group",
                text="Customers",
                category_orders={
                    "Contract": contract_order,
                    "Churn_Label": [
                        "Stayed",
                        "Churned"
                    ]
                },
                title="Customer Churn by Contract Type"
            )

            fig2.update_traces(
                textposition="outside"
            )

            fig2.update_layout(
                height=450,
                xaxis_title="Contract Type",
                yaxis_title="Number of Customers",
                legend_title="Customer Status",
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.caption(
                "Compare customer retention and churn across "
                "different contract durations."
            )

        # ====================================================
        # CHART 3
        # CHURN RATE BY INTERNET SERVICE
        # ====================================================

        st.subheader("🌐 3. Churn Rate by Internet Service")

        if (
            "InternetService" in dashboard_df.columns
            and "Churn_Label" in dashboard_df.columns
        ):

            internet_data = (
                dashboard_df[
                    dashboard_df["Churn_Label"]
                    .isin(["Churned", "Stayed"])
                ]
                .groupby("InternetService")
                .agg(
                    Total_Customers=("Churn_Label", "size"),
                    Churned_Customers=(
                        "Churn_Label",
                        lambda x: (x == "Churned").sum()
                    )
                )
                .reset_index()
            )

            internet_data["Churn_Rate"] = (
                internet_data["Churned_Customers"]
                / internet_data["Total_Customers"]
                * 100
            )

            internet_data = internet_data.sort_values(
                "Churn_Rate",
                ascending=False
            )

            fig3 = px.bar(
                internet_data,
                x="InternetService",
                y="Churn_Rate",
                text="Churn_Rate",
                title="Churn Rate by Internet Service",
                labels={
                    "InternetService": "Internet Service",
                    "Churn_Rate": "Churn Rate (%)"
                }
            )

            fig3.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig3.update_layout(
                height=450,
                yaxis_title="Churn Rate (%)",
                xaxis_title="Internet Service",
                yaxis=dict(
                    range=[
                        0,
                        max(
                            internet_data["Churn_Rate"].max() * 1.15,
                            10
                        )
                    ]
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig3,
                use_container_width=True
            )

            st.caption(
                "Churn rate is calculated as churned customers "
                "divided by total customers within each service group."
            )

        # ====================================================
        # CHART 4
        # CHURN RATE BY PAYMENT METHOD
        # ====================================================

        st.subheader("💳 4. Churn Rate by Payment Method")

        if (
            "PaymentMethod" in dashboard_df.columns
            and "Churn_Label" in dashboard_df.columns
        ):

            payment_data = (
                dashboard_df[
                    dashboard_df["Churn_Label"]
                    .isin(["Churned", "Stayed"])
                ]
                .groupby("PaymentMethod")
                .agg(
                    Total_Customers=("Churn_Label", "size"),
                    Churned_Customers=(
                        "Churn_Label",
                        lambda x: (x == "Churned").sum()
                    )
                )
                .reset_index()
            )

            payment_data["Churn_Rate"] = (
                payment_data["Churned_Customers"]
                / payment_data["Total_Customers"]
                * 100
            )

            payment_data = payment_data.sort_values(
                "Churn_Rate",
                ascending=True
            )

            fig4 = px.bar(
                payment_data,
                x="Churn_Rate",
                y="PaymentMethod",
                orientation="h",
                text="Churn_Rate",
                title="Churn Rate by Payment Method",
                labels={
                    "PaymentMethod": "Payment Method",
                    "Churn_Rate": "Churn Rate (%)"
                }
            )

            fig4.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig4.update_layout(
                height=500,
                xaxis_title="Churn Rate (%)",
                yaxis_title="Payment Method",
                xaxis=dict(
                    range=[
                        0,
                        max(
                            payment_data["Churn_Rate"].max() * 1.15,
                            10
                        )
                    ]
                ),
                margin=dict(
                    l=20,
                    r=60,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig4,
                use_container_width=True
            )

            st.caption(
                "Payment methods are ranked by their customer "
                "churn rate."
            )

        # ====================================================
        # CHART 5
        # CHURN RATE BY TENURE GROUP
        # ====================================================

        st.subheader("📈 5. Churn Rate by Tenure Group")

        if (
            "tenure" in dashboard_df.columns
            and "Churn_Label" in dashboard_df.columns
        ):

            tenure_data = dashboard_df.copy()

            # ------------------------------------------------
            # Create ordered tenure groups
            # ------------------------------------------------

            tenure_data["TenureGroup"] = pd.cut(
                tenure_data["tenure"],
                bins=[
                    -1,
                    12,
                    24,
                    48,
                    60,
                    np.inf
                ],
                labels=[
                    "0-12 months",
                    "13-24 months",
                    "25-48 months",
                    "49-60 months",
                    "60+ months"
                ],
                ordered=True
            )

            # ------------------------------------------------
            # Calculate churn rate
            # ------------------------------------------------

            tenure_churn = (
                tenure_data[
                    tenure_data["Churn_Label"]
                    .isin(["Churned", "Stayed"])
                ]
                .groupby(
                    "TenureGroup",
                    observed=False
                )
                .agg(
                    Total_Customers=("Churn_Label", "size"),
                    Churned_Customers=(
                        "Churn_Label",
                        lambda x: (x == "Churned").sum()
                    )
                )
                .reset_index()
            )

            tenure_churn["Churn_Rate"] = (
                tenure_churn["Churned_Customers"]
                / tenure_churn["Total_Customers"]
                * 100
            )

            # ------------------------------------------------
            # Line chart
            # ------------------------------------------------

            fig5 = px.line(
                tenure_churn,
                x="TenureGroup",
                y="Churn_Rate",
                markers=True,
                text="Churn_Rate",
                title="Churn Rate Across Customer Tenure"
            )

            fig5.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="top center"
            )

            fig5.update_layout(
                height=450,
                xaxis_title="Tenure Group",
                yaxis_title="Churn Rate (%)",
                yaxis=dict(
                    range=[
                        0,
                        max(
                            tenure_churn["Churn_Rate"].max() * 1.20,
                            10
                        )
                    ]
                ),
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            st.plotly_chart(
                fig5,
                use_container_width=True
            )

            st.caption(
                "This chart helps identify which customer tenure "
                "groups have the highest churn risk."
            )

        # ====================================================
        # BUSINESS INSIGHTS
        # ====================================================

        st.markdown("---")

        st.subheader("💡 Business Insights")

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:

            st.markdown(
                f"""
                **Customer Base**

                - Total customers: **{total_customers:,}**
                - Churned customers: **{churned:,}**
                - Retained customers: **{stayed:,}**
                - Overall churn rate: **{churn_rate:.2f}%**
                """
            )

        with insight_col2:

            st.markdown(
                f"""
                **Customer Economics**

                - Average monthly charge:
                  **${avg_monthly_charges:,.2f}**
                - Average customer tenure:
                  **{avg_tenure:.1f} months**
                - Model status:
                  **{"Loaded ✅" if model else "Not Found ❌"}**
                """
            )

    # ========================================================
    # PROJECT PIPELINE
    # ========================================================

    st.markdown("---")

    st.subheader("🔄 Machine Learning Pipeline")

    cols = st.columns(6)

    pipeline_steps = [
        ("1️⃣", "Data"),
        ("2️⃣", "Cleaning"),
        ("3️⃣", "EDA"),
        ("4️⃣", "Feature Engineering"),
        ("5️⃣", "ML Model"),
        ("6️⃣", "Prediction")
    ]

    for col, (number, name) in zip(
        cols,
        pipeline_steps
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>{number}</h3>
                    <strong>{name}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
       
# ============================================================
# CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer information below to estimate "
        "the likelihood of customer churn."
    )

    if model is None:

        st.error(
            "❌ Trained model not found."
        )

        st.info(
            f"Expected model directory:\n\n{MODEL_DIR}"
        )

        st.stop()

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col3:

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    # --------------------------------------------------------
    # CONTRACT / BILLING
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("💳 Contract & Billing")

    col1, col2, col3 = st.columns(3)

    with col1:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=monthly_charges * tenure,
            step=10.0
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.markdown("---")

    predict_button = st.button(
        "🚀 Predict Customer Churn",
        use_container_width=True,
        type="primary"
    )

    if predict_button:

        customer = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }]
        )

        try:

            prediction, probability = predict_customer(
                model,
                customer
            )

            # ------------------------------------------------
            # HANDLE PREDICTION
            # ------------------------------------------------

            prediction_string = (
                str(prediction)
                .strip()
                .lower()
            )

            churn_prediction = prediction_string in [
                "yes",
                "1",
                "true",
                "churn",
                "churned"
            ]

            if probability is not None:

                if len(probability) == 2:

                    churn_probability = float(
                        probability[1]
                    )

                else:

                    churn_probability = float(
                        np.max(probability)
                    )

            else:

                churn_probability = None

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            if churn_prediction:

                st.markdown(
                    """
                    <div class="prediction-box">
                        <h2>⚠️ High Churn Risk</h2>
                        <p>
                            This customer is predicted to churn.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="prediction-box">
                        <h2>✅ Low Churn Risk</h2>
                        <p>
                            This customer is predicted to stay.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if churn_probability is not None:

                st.metric(
                    "Estimated Churn Probability",
                    f"{churn_probability * 100:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            churn_probability,
                            0.0
                        ),
                        1.0
                    )
                )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            st.markdown("---")

            st.subheader(
                "💡 Business Recommendation"
            )

            if churn_prediction:

                st.warning(
                    """
                    **Recommended actions:**

                    - Contact the customer with a retention offer.
                    - Review their monthly charges.
                    - Consider contract upgrade incentives.
                    - Offer technical support if applicable.
                    - Provide personalized discounts.
                    - Monitor this customer closely.
                    """
                )

            else:

                st.success(
                    """
                    **Recommended actions:**

                    - Continue normal customer engagement.
                    - Consider loyalty rewards.
                    - Promote additional services.
                    - Monitor customer satisfaction.
                    """
                )

            # ------------------------------------------------
            # SAVE PREDICTION
            # ------------------------------------------------

            st.session_state.prediction_history.append(
                {
                    "Prediction": (
                        "Churn"
                        if churn_prediction
                        else "Stay"
                    ),
                    "Churn Probability": (
                        churn_probability
                        if churn_probability is not None
                        else np.nan
                    ),
                    "Tenure": tenure,
                    "Monthly Charges": monthly_charges
                }
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.exception(e)

            st.info(
                """
                This usually happens when the columns supplied
                by the Streamlit application do not exactly match
                the columns expected by your trained pipeline.
                """
            )


# ============================================================
# DATASET
# ============================================================

elif page == "📊 Dataset":

    st.title("📊 Telecom Customer Dataset")

    if dataset is None:

        st.error(
            "Dataset not found in the data directory."
        )

        st.info(
            f"Place your CSV inside:\n\n{DATA_DIR}"
        )

    else:

        st.success(
            f"Dataset loaded successfully: "
            f"{dataset.shape[0]:,} rows × "
            f"{dataset.shape[1]:,} columns"
        )

        # ----------------------------------------------------
        # DATASET PREVIEW
        # ----------------------------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(
            dataset.head(20),
            use_container_width=True
        )

        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Rows",
                f"{dataset.shape[0]:,}"
            )

        with col2:

            st.metric(
                "Columns",
                f"{dataset.shape[1]:,}"
            )

        with col3:

            missing_values = int(
                dataset.isna().sum().sum()
            )

            st.metric(
                "Missing Values",
                f"{missing_values:,}"
            )

        # ----------------------------------------------------
        # COLUMN INFORMATION
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("📋 Column Information")

        column_info = pd.DataFrame(
            {
                "Column": dataset.columns,
                "Data Type": [
                    str(dtype)
                    for dtype in dataset.dtypes
                ],
                "Missing Values": [
                    int(
                        dataset[col].isna().sum()
                    )
                    for col in dataset.columns
                ],
                "Unique Values": [
                    int(
                        dataset[col].nunique()
                    )
                    for col in dataset.columns
                ]
            }
        )

        st.dataframe(
            column_info,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.markdown(
        """
        ## Telecom Customer Churn Prediction

        This project uses Machine Learning to identify telecom
        customers who are likely to discontinue their services.

        ### 🎯 Business Problem

        Customer churn is an important business problem for telecom
        companies because acquiring a new customer can be more
        expensive than retaining an existing customer.

        The objective of this project is to:

        - Analyze customer behavior
        - Identify important churn drivers
        - Clean and preprocess customer data
        - Engineer useful features
        - Train multiple machine learning models
        - Tune the best-performing model
        - Predict customer churn
        - Provide actionable business recommendations

        ### 🛠️ Technologies Used

        **Python**

        - Pandas
        - NumPy
        - Scikit-learn
        - Matplotlib
        - Seaborn
        - Plotly
        - SHAP
        - Joblib

        **Machine Learning**

        - Logistic Regression
        - Random Forest
        - Gradient Boosting
        - Model Evaluation
        - Cross Validation
        - RandomizedSearchCV
        - ROC-AUC
        - Confusion Matrix
        - SHAP Explainability

        **Deployment**

        - Streamlit

        ### 🔄 Project Workflow

        `Raw Data`

        ↓

        `Data Cleaning`

        ↓

        `Exploratory Data Analysis`

        ↓

        `Feature Engineering`

        ↓

        `Train/Test Split`

        ↓

        `Preprocessing Pipeline`

        ↓

        `Multiple ML Models`

        ↓

        `Model Evaluation`

        ↓

        `Hyperparameter Tuning`

        ↓

        `SHAP Explainability`

        ↓

        `Final Model`

        ↓

        `Streamlit Deployment`

        ### 👨‍💻 Project Objective

        The final application allows business users to enter
        customer information and receive a machine-learning-based
        churn prediction along with an estimated probability and
        recommended retention actions.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Telecom Customer Churn Prediction • "
    "Machine Learning Portfolio Project"
)