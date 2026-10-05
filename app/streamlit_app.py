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
    except Exception as e:
        return None


@st.cache_data
def load_dataset():
    """
    Try to locate and load the telecom churn dataset.
    """
    possible_files = [
        os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
        os.path.join(DATA_DIR, "telco_customer_churn.csv"),
        os.path.join(DATA_DIR, "Telco-Customer-Churn.csv"),
        os.path.join(DATA_DIR, "telco-churn.csv"),
        os.path.join(DATA_DIR, "cleaned_telco_churn.csv"),
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
        model_path = os.path.join(MODEL_DIR, model_name)

        if os.path.exists(model_path):
            return model_path

    # If no preferred model is found,
    # search for any .pkl model.
    if os.path.exists(MODEL_DIR):

        model_files = [
            file for file in os.listdir(MODEL_DIR)
            if file.endswith(".pkl")
        ]

        if model_files:
            return os.path.join(MODEL_DIR, model_files[0])

    return None


def clean_input_data(df):
    """
    Basic cleaning for prediction input.

    IMPORTANT:
    If your saved model is a complete sklearn Pipeline,
    avoid doing transformations here that are already handled
    inside the pipeline.
    """

    df = df.copy()

    # Convert TotalCharges if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
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
        probability = model.predict_proba(customer_df)[0]

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

    st.title("📡 Telecom Customer Churn Prediction")

    st.markdown(
        """
        ### Predict customer churn using Machine Learning

        This application analyzes telecom customer information and
        predicts whether a customer is likely to **Churn** or **Stay**.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    if dataset is not None:

        total_customers = len(dataset)

        if "Churn" in dataset.columns:

            churn_values = (
                dataset["Churn"]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            churned = churn_values.isin(
                ["yes", "1", "true", "churn"]
            ).sum()

            churn_rate = (
                churned / total_customers * 100
                if total_customers > 0
                else 0
            )

        else:
            churned = 0
            churn_rate = 0

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "👥 Customers",
                f"{total_customers:,}"
            )

        with col2:
            st.metric(
                "⚠️ Churned",
                f"{churned:,}"
            )

        with col3:
            st.metric(
                "📉 Churn Rate",
                f"{churn_rate:.2f}%"
            )

        with col4:
            st.metric(
                "🤖 Model",
                "Loaded" if model else "Not Found"
            )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("👥 Dataset", "Not Found")

        with col2:
            st.metric("🤖 Model", "Loaded" if model else "Not Found")

        with col3:
            st.metric("📊 Status", "Ready")

    st.markdown("---")

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    if dataset is not None and "Churn" in dataset.columns:

        st.subheader("📊 Churn Distribution")

        churn_counts = dataset["Churn"].value_counts()

        col1, col2 = st.columns(2)

        with col1:

            st.bar_chart(churn_counts)

        with col2:

            st.dataframe(
                churn_counts.rename("Customers"),
                use_container_width=True
            )

    # --------------------------------------------------------
    # PROJECT PIPELINE
    # --------------------------------------------------------

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

    for col, (number, name) in zip(cols, pipeline_steps):

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
        "Enter customer information below to estimate the likelihood "
        "of customer churn."
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

            prediction_string = str(prediction).strip().lower()

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
                        <p>This customer is predicted to churn.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="prediction-box">
                        <h2>✅ Low Churn Risk</h2>
                        <p>This customer is predicted to stay.</p>
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
                    min(max(churn_probability, 0.0), 1.0)
                )

            # ------------------------------------------------
            # RECOMMENDATION
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("💡 Business Recommendation")

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

            # Save prediction
            st.session_state.prediction_history.append(
                {
                    "Prediction": (
                        "Churn" if churn_prediction else "Stay"
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
                This usually happens when the columns supplied by the
                Streamlit application do not exactly match the columns
                expected by your trained pipeline.
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
            f"Dataset loaded successfully: {dataset.shape[0]:,} rows × "
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
                    int(dataset[col].isna().sum())
                    for col in dataset.columns
                ],
                "Unique Values": [
                    int(dataset[col].nunique())
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

        This project uses Machine Learning to identify telecom customers
        who are likely to discontinue their services.

        ### 🎯 Business Problem

        Customer churn is an important business problem for telecom
        companies because acquiring a new customer can be more expensive
        than retaining an existing customer.

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

        The final application allows business users to enter customer
        information and receive a machine-learning-based churn prediction
        along with an estimated probability and recommended retention
        actions.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Telecom Customer Churn Prediction • Machine Learning Portfolio Project"
)