import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ChurnSense | Telco Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 ChurnSense")

st.sidebar.markdown(
    """
    ### Telco Customer Churn Prediction

    Use the pages in the sidebar to:

    - 🏠 View the project dashboard
    - 🔮 Predict customer churn
    - 📁 Perform batch predictions
    - 📈 Explore churn analytics
    """
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine Learning project for predicting "
    "telecom customer churn."
)


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.title("📊 ChurnSense")

st.subheader("Telco Customer Churn Prediction")

st.markdown(
    """
    **ChurnSense** is an end-to-end machine learning project
    designed to identify customers who are at risk of leaving
    a telecommunications company.
    """
)


# ============================================================
# PROJECT OBJECTIVE
# ============================================================

st.markdown("## 🎯 Business Objective")

st.write(
    """
    Customer churn can significantly affect a telecom company's
    revenue. The objective of this project is to analyze customer
    behavior and build a machine learning model that estimates
    the probability of customer churn.
    """
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown("## 📂 Dataset")

if DATA_PATH.exists():

    try:
        df = pd.read_csv(DATA_PATH)

        total_customers = len(df)

        if "Churn" in df.columns:

            churned_customers = (
                df["Churn"]
                .astype(str)
                .str.strip()
                .eq("Yes")
                .sum()
            )

            churn_rate = (
                churned_customers / total_customers * 100
                if total_customers > 0
                else 0
            )

        else:
            churned_customers = 0
            churn_rate = 0

        if "MonthlyCharges" in df.columns:
            avg_monthly_charges = df["MonthlyCharges"].mean()
        else:
            avg_monthly_charges = 0

        if "tenure" in df.columns:
            avg_tenure = df["tenure"].mean()
        else:
            avg_tenure = 0

        # ====================================================
        # KPI CARDS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "👥 Total Customers",
                f"{total_customers:,}"
            )

        with col2:
            st.metric(
                "⚠️ Churned Customers",
                f"{churned_customers:,}"
            )

        with col3:
            st.metric(
                "📉 Churn Rate",
                f"{churn_rate:.2f}%"
            )

        with col4:
            st.metric(
                "💰 Avg Monthly Charges",
                f"${avg_monthly_charges:.2f}"
            )

        st.markdown("---")

        # ====================================================
        # DATASET PREVIEW
        # ====================================================

        st.markdown("## 🔍 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Unable to load dataset: {e}"
        )

else:

    st.warning(
        "Dataset not found. Please place the Telco Customer "
        "Churn CSV inside the data folder."
    )


# ============================================================
# PROJECT WORKFLOW
# ============================================================

st.markdown("## 🔄 Project Workflow")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        ### 1️⃣ Data Analysis

        - Data cleaning
        - Missing values
        - Exploratory Data Analysis
        - Business insights
        """
    )

with col2:
    st.markdown(
        """
        ### 2️⃣ Preprocessing

        - Feature engineering
        - Numerical scaling
        - Categorical encoding
        - Train/test split
        """
    )

with col3:
    st.markdown(
        """
        ### 3️⃣ Machine Learning

        - Logistic Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        - XGBoost
        """
    )

with col4:
    st.markdown(
        """
        ### 4️⃣ Deployment

        - Model evaluation
        - SHAP explainability
        - Streamlit application
        - Batch prediction
        """
    )


# ============================================================
# HOW TO USE
# ============================================================

st.markdown("## 🚀 How to Use the Application")

st.write(
    """
    **Predict:** Enter information about an individual customer
    and generate a churn prediction.

    **Batch:** Upload a CSV file containing multiple customers
    and generate predictions for all customers.

    **Analytics:** Explore customer churn patterns and business
    insights.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "ChurnSense | Telco Customer Churn Prediction | "
    "Data Analytics & Machine Learning Project"
)