"""
ChurnSense - Telco Customer Churn Prediction
============================================
Structure follows the Session 9 final-presentation flow:

  1. Problem Statement  (what is churn, why ChurnSense matters)
  2. EDA Highlights     (5 best charts + key findings, computed from the data)
  3. Model Approach     (5 models, winner, comparison)
  4. Live App           (Dashboard > Predict > Batch > Analytics)
  5. Learnings

Run:  streamlit run app/streamlit_app.py

Needs: streamlit, pandas, numpy, plotly, scikit-learn, joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# CONFIG  (edit the text here, not all over the file)
# --------------------------------------------------------------------------
st.set_page_config(page_title="ChurnSense", page_icon="📡", layout="wide")

APP_DIR = Path(__file__).resolve().parent
ROOTS = [APP_DIR, APP_DIR.parent]          # works for app/streamlit_app.py or root

RED, TEAL, AMBER, GREY = "#EF4444", "#14B8A6", "#F59E0B", "#94A3B8"
COLOR_MAP = {"Yes": RED, "No": TEAL}

# Edit these so they reflect YOUR project (they appear on the last page)
MODELS_TRAINED = [
    ("Logistic Regression", "Baseline - simple, fast, interpretable"),
    ("Decision Tree", "Explainable rules a manager can read"),
    ("Random Forest", "Ensemble of trees, robust to noise"),
    ("Gradient Boosting", "Sequential trees that fix earlier mistakes"),
    ("XGBoost", "Industry standard for tabular data"),
]
LEARNINGS = [
    ("Hardest part", "Write here what was hardest, e.g. handling class imbalance "
                     "or fixing paths for deployment."),
    ("Biggest lesson", "Write here, e.g. accuracy is misleading - recall matters "
                       "more when the goal is catching churners."),
    ("What's next", "Add SHAP explanations per customer, an automated retraining "
                    "job, and a CRM integration to trigger retention offers."),
]

# --------------------------------------------------------------------------
# STYLE
# --------------------------------------------------------------------------
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hero {{
    padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 60%, #0D9488 100%);
    color: #fff;
}}
.hero h1 {{ margin: 0; font-size: 2.6rem; color: #fff; }}
.hero p  {{ margin: .4rem 0 0; font-size: 1.1rem; opacity: .9; color: #fff; }}
.card {{
    border: 1px solid rgba(148,163,184,.35); border-radius: 14px;
    padding: 1rem 1.2rem; background: rgba(148,163,184,.08); height: 100%;
}}
.card h4 {{ margin: 0 0 .4rem; }}
.kpi {{
    border-radius: 14px; padding: 1rem 1.1rem;
    border: 1px solid rgba(148,163,184,.35); background: rgba(148,163,184,.08);
}}
.kpi .label {{ font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em; }}
.kpi .value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.2; }}
.kpi .sub   {{ font-size: .8rem; opacity: .65; }}
.badge {{
    display: inline-block; padding: .35rem 1rem; border-radius: 999px;
    font-weight: 700; color: #fff; font-size: 1.05rem;
}}
.finding {{
    border-left: 4px solid {TEAL}; padding: .6rem 1rem; margin: .5rem 0;
    background: rgba(20,184,166,.08); border-radius: 0 10px 10px 0;
}}
.step {{ font-size: 2rem; }}
</style>
""",
    unsafe_allow_html=True,
)


def kpi(label, value, sub=""):
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def finding(text):
    st.markdown(f'<div class="finding">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text="", template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# --------------------------------------------------------------------------
# LOADERS
# --------------------------------------------------------------------------
def _find_file(patterns):
    for root in ROOTS:
        for pat in patterns:
            hits = sorted(root.glob(pat))
            if hits:
                return hits[0]
    return None


@st.cache_data(show_spinner="Loading data...")
def load_data():
    path = _find_file(["data/**/*hurn*.csv", "data/**/*.csv", "*hurn*.csv"])
    if path is None:
        return None
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorLabel"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    path = _find_file(["models/churn_model_v1.pkl", "models/*.pkl", "*.pkl"])
    if path is None:
        return None
    obj = joblib.load(path)
    if isinstance(obj, dict):  # in case you saved {"model": pipeline, ...}
        obj = obj.get("model") or obj.get("pipeline") or next(iter(obj.values()))
    return obj


df = load_data()
model = load_model()


# --------------------------------------------------------------------------
# MODEL HELPERS
# --------------------------------------------------------------------------
SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Recreate the Session-3 engineered features IF the model expects them.
    >>> Make these match the logic in your src/preprocessing.py <<<"""
    X = X.copy()
    if "tenure_group" not in X.columns:
        X["tenure_group"] = pd.cut(X["tenure"], [-1, 12, 24, 48, 72],
                                   labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    if "num_services" not in X.columns:
        X["num_services"] = sum((X[c] == "Yes").astype(int) for c in SERVICE_COLS if c in X)
    return X


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Drop non-features and align columns to what the saved pipeline expects."""
    X = X.drop(columns=["customerID", "Churn", "churn_flag", "SeniorLabel"], errors="ignore")
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return X
    X = add_engineered_features(X)
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return X[list(expected)]


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare(X))[:, 1]


def risk_level(p: float):
    if p < 0.35:
        return "LOW RISK", TEAL
    if p < 0.65:
        return "MEDIUM RISK", AMBER
    return "HIGH RISK", RED


@st.cache_data(show_spinner="Evaluating model on the hold-out test set...")
def evaluate(_model, data: pd.DataFrame):
    """Same stratified 80/20 split as Session 3 (random_state=42)."""
    X = data.drop(columns=["Churn", "churn_flag", "SeniorLabel", "customerID"], errors="ignore")
    y = data["churn_flag"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_te_p = prepare(X_te)
    prob = _model.predict_proba(X_te_p)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, prob)
    metrics = {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "ROC-AUC": roc_auc_score(y_te, prob),
    }
    imp = permutation_importance(_model, X_te_p, y_te, scoring="roc_auc",
                                 n_repeats=5, random_state=42)
    imp_df = (pd.DataFrame({"Feature": X_te_p.columns, "Importance": imp.importances_mean})
              .sort_values("Importance").tail(10))
    return metrics, confusion_matrix(y_te, pred), (fpr, tpr), imp_df, (y_te.values, prob)


def need_data():
    st.error("Dataset not found. Put the Telco CSV in a `data/` folder next to the app "
             "(for example `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`).")
    st.stop()


def need_model():
    st.error("Model not found. Commit your trained pipeline as `models/churn_model_v1.pkl`.")
    st.stop()


# --------------------------------------------------------------------------
# PAGE 1 - PROBLEM STATEMENT
# --------------------------------------------------------------------------
def page_problem():
    st.markdown(
        '<div class="hero"><h1>📡 ChurnSense</h1>'
        "<p>Predict which telecom customers are about to leave - "
        "and act before they do.</p></div>",
        unsafe_allow_html=True,
    )

    st.subheader("🎯 Problem Statement")
    st.markdown(
        "A telecom company loses revenue every time a subscriber cancels. "
        "The business usually finds out **after** the customer has left, when it is too late "
        "to do anything. **ChurnSense** uses machine learning on customer account data to give "
        "every customer a *churn probability*, so retention teams can focus their calls, "
        "offers and discounts on the people most likely to leave."
    )

    c1, c2 = st.columns(2)
    with c1:
        card("❓ What is churn?",
             "<b>Customer churn</b> means a customer stops using a company's service. "
             "<br><br><code>Churn rate = customers lost ÷ total customers</code>"
             "<br><br>In this dataset a customer is marked <b>Churn = Yes</b> if they left "
             "within the last month.")
    with c2:
        card("💰 Why does ChurnSense matter?",
             "Winning a new customer typically costs several times more than keeping an "
             "existing one. Even a small drop in churn protects a large amount of "
             "recurring revenue. A model that flags at-risk customers early turns "
             "<i>guesswork</i> into a <b>ranked call list</b>.")

    if df is not None:
        st.markdown("&nbsp;")
        lost = df.loc[df.churn_flag == 1, "MonthlyCharges"].sum()
        k1, k2, k3, k4 = st.columns(4)
        with k1: kpi("Customers analysed", f"{len(df):,}", "rows in dataset")
        with k2: kpi("Churn rate", f"{df.churn_flag.mean():.1%}", "customers who left")
        with k3: kpi("Monthly revenue lost", f"${lost:,.0f}", "from churned customers")
        with k4: kpi("Yearly revenue lost", f"${lost * 12:,.0f}", "if the loss continues")

    st.markdown("&nbsp;")
    st.subheader("🧭 How the project works")
    cols = st.columns(5)
    steps = [("🗂️", "Data", "7,043 customers, 21 columns"),
             ("📊", "EDA", "Find what drives churn"),
             ("⚙️", "Preprocess", "Clean, encode, scale"),
             ("🤖", "Model", "5 models, best one tuned"),
             ("🚀", "Deploy", "Live Streamlit app")]
    for col, (icon, title, text) in zip(cols, steps):
        with col:
            card(f"{icon} {title}", text)


# --------------------------------------------------------------------------
# PAGE 2 - EDA HIGHLIGHTS
# --------------------------------------------------------------------------
def rate_by(col, data=None):
    data = df if data is None else data
    return data.groupby(col)["churn_flag"].mean().mul(100).round(1)


def page_eda():
    if df is None:
        need_data()
    st.title("📊 EDA Highlights")
    st.caption("Five charts that explain who churns - every number below is computed live from the data.")

    overall = df.churn_flag.mean() * 100
    contract = rate_by("Contract")
    fiber = rate_by("InternetService").get("Fiber optic", np.nan)
    echeck = rate_by("PaymentMethod").get("Electronic check", np.nan)
    med_churn = df.loc[df.churn_flag == 1, "tenure"].median()
    med_stay = df.loc[df.churn_flag == 0, "tenure"].median()
    first_year = df.loc[df.tenure <= 12, "churn_flag"].mean() * 100
    mc_churn = df.loc[df.churn_flag == 1, "MonthlyCharges"].mean()
    mc_stay = df.loc[df.churn_flag == 0, "MonthlyCharges"].mean()

    # ---- Chart 1
    st.subheader("1 · The target is imbalanced")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Customers"]
        fig = px.pie(counts, names="Churn", values="Customers", hole=0.55,
                     color="Churn", color_discrete_map=COLOR_MAP,
                     title="Churn distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        finding(f"Only <b>{overall:.1f}%</b> of customers churned, so the classes are "
                f"imbalanced (about {100 - overall:.0f} : {overall:.0f}).")
        finding("A model that always says <i>'No churn'</i> would be "
                f"<b>{100 - overall:.0f}% accurate</b> and still useless. That is why we judge "
                "the model on <b>Recall</b> and <b>ROC-AUC</b>, not accuracy alone.")
        finding("We handled this with <code>class_weight='balanced'</code> and stratified splits.")

    # ---- Chart 2
    st.subheader("2 · Contract type is the strongest signal")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        d = contract.reset_index()
        d.columns = ["Contract", "Churn rate (%)"]
        fig = px.bar(d, x="Contract", y="Churn rate (%)", text="Churn rate (%)",
                     color="Churn rate (%)", color_continuous_scale=["#14B8A6", "#F59E0B", "#EF4444"],
                     title="Churn rate by contract type")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        finding(f"Month-to-month customers churn at <b>{contract.get('Month-to-month', np.nan):.1f}%</b> "
                f"versus <b>{contract.get('Two year', np.nan):.1f}%</b> on two-year contracts.")
        finding("<b>Action:</b> move month-to-month customers to annual plans with a discount or perk.")

    # ---- Chart 3
    st.subheader("3 · New customers leave first")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.histogram(df, x="tenure", color="Churn", nbins=36, barmode="overlay",
                           opacity=0.75, color_discrete_map=COLOR_MAP,
                           title="Tenure (months) by churn")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        finding(f"Churned customers have a median tenure of <b>{med_churn:.0f} months</b> "
                f"vs <b>{med_stay:.0f} months</b> for loyal ones.")
        finding(f"Customers in their first year churn at <b>{first_year:.1f}%</b>.")
        finding("<b>Action:</b> invest in onboarding and early check-ins during months 1-12.")

    # ---- Chart 4
    st.subheader("4 · Higher bills, higher churn")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn", points=False,
                     color_discrete_map=COLOR_MAP, title="Monthly charges by churn")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        finding(f"Churned customers pay on average <b>${mc_churn:.0f}</b> per month versus "
                f"<b>${mc_stay:.0f}</b> for retained customers.")
        finding(f"Fiber optic users churn at <b>{fiber:.1f}%</b> and electronic-check payers "
                f"at <b>{echeck:.1f}%</b> - both well above the {overall:.1f}% average.")
        finding("<b>Action:</b> review fiber pricing and push auto-pay to reduce friction.")

    # ---- Chart 5
    st.subheader("5 · Tech support protects internet customers")
    c1, c2 = st.columns([1.2, 1])
    net = df[df["InternetService"] != "No"]
    g = (net.groupby(["InternetService", "TechSupport"])["churn_flag"].mean().mul(100)
         .round(1).reset_index().rename(columns={"churn_flag": "Churn rate (%)"}))
    with c1:
        fig = px.bar(g, x="InternetService", y="Churn rate (%)", color="TechSupport",
                     barmode="group", text="Churn rate (%)",
                     color_discrete_map={"Yes": TEAL, "No": RED},
                     title="Churn rate by internet service and tech support")
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with c2:
        ts = rate_by("TechSupport")
        finding(f"Customers <b>without</b> tech support churn at <b>{ts.get('No', np.nan):.1f}%</b>; "
                f"with it only <b>{ts.get('Yes', np.nan):.1f}%</b>.")
        finding("<b>Action:</b> bundle free tech support for at-risk fiber and DSL customers.")

    with st.expander("➕ Bonus: correlation heatmap"):
        num = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "churn_flag"]].corr()
        fig = px.imshow(num.round(2), text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Correlation of numeric features")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.success("**Key takeaway:** the typical churner is a new, month-to-month customer on a "
               "high monthly bill, using fiber, paying by electronic check, with no tech support.")


# --------------------------------------------------------------------------
# PAGE 3 - MODEL APPROACH
# --------------------------------------------------------------------------
def page_model():
    st.title("🤖 Model Approach")
    st.markdown("Five models were trained inside full scikit-learn **Pipelines** "
                "(scaler + one-hot encoder + model) and compared with "
                "**5-fold stratified cross-validation**.")
    cols = st.columns(5)
    for col, (name, why) in zip(cols, MODELS_TRAINED):
        with col:
            card(name, why)

    st.markdown("&nbsp;")
    comp_path = _find_file(["reports/model_comparison.csv", "reports/*comparison*.csv"])
    if comp_path is not None:
        st.subheader("Model comparison")
        st.dataframe(pd.read_csv(comp_path), use_container_width=True, hide_index=True)
    else:
        st.info("Tip: save your Session-4 comparison table with "
                "`comparison_df.to_csv('reports/model_comparison.csv', index=False)` "
                "and it will appear here automatically.")

    if model is not None and df is not None:
        m, *_ = evaluate(model, df)
        st.subheader("🏆 Final tuned model - hold-out test set")
        cols = st.columns(5)
        for col, (k, v) in zip(cols, m.items()):
            with col:
                kpi(k, f"{v:.3f}")
        if m["ROC-AUC"] >= 0.80:
            st.success(f"ROC-AUC {m['ROC-AUC']:.3f} meets the project target of ≥ 0.80.")


# --------------------------------------------------------------------------
# PAGE 4 - LIVE APP
# --------------------------------------------------------------------------
def tab_dashboard():
    if df is None:
        need_data()
    st.subheader("Business dashboard")

    with st.expander("🔎 Filters", expanded=False):
        f1, f2, f3 = st.columns(3)
        contracts = f1.multiselect("Contract", sorted(df.Contract.unique()), sorted(df.Contract.unique()))
        internet = f2.multiselect("Internet service", sorted(df.InternetService.unique()),
                                  sorted(df.InternetService.unique()))
        tmin, tmax = f3.slider("Tenure (months)", 0, int(df.tenure.max()), (0, int(df.tenure.max())))
    d = df[df.Contract.isin(contracts) & df.InternetService.isin(internet)
           & df.tenure.between(tmin, tmax)]
    if d.empty:
        st.warning("No customers match these filters.")
        return

    at_risk = d.loc[d.churn_flag == 1, "MonthlyCharges"].sum()
    k = st.columns(5)
    with k[0]: kpi("Customers", f"{len(d):,}")
    with k[1]: kpi("Churn rate", f"{d.churn_flag.mean():.1%}")
    with k[2]: kpi("Avg monthly bill", f"${d.MonthlyCharges.mean():.0f}")
    with k[3]: kpi("Avg tenure", f"{d.tenure.mean():.0f} mo")
    with k[4]: kpi("Monthly revenue lost", f"${at_risk:,.0f}")
    st.markdown("&nbsp;")

    c1, c2 = st.columns(2)
    with c1:
        r = d.groupby("Contract")["churn_flag"].mean().mul(100).reset_index()
        fig = px.bar(r, x="Contract", y="churn_flag", title="Churn rate by contract (%)",
                     color_discrete_sequence=[RED], labels={"churn_flag": "Churn rate (%)"})
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with c2:
        r = d.groupby("PaymentMethod")["churn_flag"].mean().mul(100).sort_values().reset_index()
        fig = px.bar(r, x="churn_flag", y="PaymentMethod", orientation="h",
                     title="Churn rate by payment method (%)",
                     color_discrete_sequence=[AMBER], labels={"churn_flag": "Churn rate (%)"})
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        d2 = d.assign(bucket=pd.cut(d.tenure, [-1, 12, 24, 48, 72],
                                    labels=["0-12", "13-24", "25-48", "49-72"]))
        r = d2.groupby("bucket", observed=True)["churn_flag"].mean().mul(100).reset_index()
        fig = px.line(r, x="bucket", y="churn_flag", markers=True,
                      title="Churn rate by tenure group (%)", labels={"churn_flag": "Churn rate (%)", "bucket": "Tenure (months)"})
        fig.update_traces(line_color=RED)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)
    with c4:
        fig = px.scatter(d.sample(min(len(d), 1500), random_state=1), x="tenure", y="MonthlyCharges",
                         color="Churn", color_discrete_map=COLOR_MAP, opacity=0.6,
                         title="Tenure vs monthly charges")
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)


def gauge(p):
    label, color = risk_level(p)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 35], "color": "rgba(20,184,166,.25)"},
                         {"range": [35, 65], "color": "rgba(245,158,11,.25)"},
                         {"range": [65, 100], "color": "rgba(239,68,68,.25)"}]},
        title={"text": "Churn probability"}))
    fig.update_layout(height=290, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def recommendations(r):
    tips = []
    if r["Contract"] == "Month-to-month":
        tips.append("Offer a discounted 1-year or 2-year contract.")
    if r["TechSupport"] == "No" and r["InternetService"] != "No":
        tips.append("Add a free 3-month tech-support trial.")
    if r["OnlineSecurity"] == "No" and r["InternetService"] != "No":
        tips.append("Bundle online security to increase stickiness.")
    if r["PaymentMethod"] == "Electronic check":
        tips.append("Encourage auto-pay (card / bank transfer) with a small credit.")
    if r["tenure"] <= 12:
        tips.append("Schedule an onboarding check-in call - first-year customers are most fragile.")
    if r["InternetService"] == "Fiber optic" and r["MonthlyCharges"] > 80:
        tips.append("Review the fiber plan price or offer a loyalty discount.")
    return tips or ["Customer looks stable - keep the relationship warm with regular updates."]


def tab_predict():
    if model is None:
        need_model()
    st.subheader("Predict churn for one customer")
    yn = ["Yes", "No"]
    with st.form("predict"):
        a, b, c = st.columns(3)
        with a:
            st.markdown("**👤 Profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", yn)
            dependents = st.selectbox("Dependents", yn)
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with b:
            st.markdown("**📞 Services**")
            phone = st.selectbox("Phone service", yn)
            lines = st.selectbox("Multiple lines", ["No", "Yes"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online security", yn[::-1])
            backup = st.selectbox("Online backup", yn[::-1])
            device = st.selectbox("Device protection", yn[::-1])
        with c:
            st.markdown("**💳 Billing**")
            tech = st.selectbox("Tech support", yn[::-1])
            tv = st.selectbox("Streaming TV", yn[::-1])
            movies = st.selectbox("Streaming movies", yn[::-1])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", yn)
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                      "Bank transfer (automatic)",
                                                      "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5)
        submitted = st.form_submit_button("🔮 Predict churn risk", use_container_width=True)

    if not submitted:
        st.info("Fill in the customer details and press **Predict**.")
        return

    # keep the dependent fields consistent with the original dataset's coding
    no_net = internet == "No"
    row = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": "No phone service" if phone == "No" else lines,
        "InternetService": internet,
        "OnlineSecurity": "No internet service" if no_net else security,
        "OnlineBackup": "No internet service" if no_net else backup,
        "DeviceProtection": "No internet service" if no_net else device,
        "TechSupport": "No internet service" if no_net else tech,
        "StreamingTV": "No internet service" if no_net else tv,
        "StreamingMovies": "No internet service" if no_net else movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": round(monthly * max(tenure, 1), 2),
    }
    try:
        p = float(predict_proba(pd.DataFrame([row]))[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    label, color = risk_level(p)
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(gauge(p), use_container_width=True)
    with right:
        st.markdown(f'<span class="badge" style="background:{color}">{label}</span>',
                    unsafe_allow_html=True)
        st.markdown(f"#### This customer has a **{p:.0%}** chance of leaving.")
        st.markdown("**Suggested retention actions**")
        for t in recommendations(row):
            st.markdown(f"- {t}")


def tab_batch():
    if model is None:
        need_model()
    st.subheader("Score many customers at once")
    st.caption("Upload a CSV with the same columns as the Telco dataset "
               "(`Churn` and `customerID` are optional).")

    if df is not None:
        sample = df.drop(columns=["Churn", "churn_flag", "SeniorLabel"]).head(10)
        st.download_button("⬇️ Download a sample template", sample.to_csv(index=False),
                           "churnsense_template.csv", "text/csv")

    up = st.file_uploader("Upload CSV", type="csv")
    if up is None:
        return
    data = pd.read_csv(up)
    if "TotalCharges" in data:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0.0)
    st.write(f"Loaded **{len(data):,}** rows.")

    try:
        prob = predict_proba(data)
    except Exception as e:
        st.error(f"Could not score this file: {e}")
        return

    out = data.copy()
    out["churn_probability"] = prob.round(4)
    out["prediction"] = np.where(prob >= 0.5, "Churn", "Stay")
    out["risk_level"] = [risk_level(x)[0].split()[0] for x in prob]

    k = st.columns(4)
    with k[0]: kpi("Customers scored", f"{len(out):,}")
    with k[1]: kpi("Predicted to churn", f"{(prob >= 0.5).sum():,}")
    with k[2]: kpi("High risk", f"{(prob >= 0.65).sum():,}")
    with k[3]: kpi("Avg probability", f"{prob.mean():.1%}")
    st.markdown("&nbsp;")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(out, x="churn_probability", nbins=30,
                           title="Distribution of churn probability",
                           color_discrete_sequence=[RED])
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with c2:
        rc = out["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).reset_index()
        rc.columns = ["Risk", "Customers"]
        fig = px.bar(rc, x="Risk", y="Customers", color="Risk", title="Customers by risk level",
                     color_discrete_map={"LOW": TEAL, "MEDIUM": AMBER, "HIGH": RED})
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)

    st.markdown("**🔥 Top 10 customers to call first**")
    show = [c for c in ["customerID", "Contract", "tenure", "MonthlyCharges",
                        "churn_probability", "risk_level"] if c in out.columns]
    st.dataframe(out.sort_values("churn_probability", ascending=False)[show].head(10),
                 use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download all predictions (CSV)", out.to_csv(index=False),
                       "churnsense_predictions.csv", "text/csv", use_container_width=True)


def tab_analytics():
    if model is None:
        need_model()
    if df is None:
        need_data()
    st.subheader("Model analytics")
    try:
        m, cm, (fpr, tpr), imp, (y_true, y_prob) = evaluate(model, df)
    except Exception as e:
        st.error(f"Could not evaluate the model: {e}")
        return

    cols = st.columns(5)
    for col, (k, v) in zip(cols, m.items()):
        with col:
            kpi(k, f"{v:.3f}")
    st.caption("Metrics on the 20% stratified hold-out set at a 0.5 threshold. "
               "Recall is the key metric: it tells us how many real churners we catch.")
    st.markdown("&nbsp;")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Model (AUC {m['ROC-AUC']:.3f})",
                                 line=dict(color=TEAL, width=3), fill="tozeroy",
                                 fillcolor="rgba(20,184,166,.15)"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess",
                                 line=dict(color=GREY, dash="dash")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c2:
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Stay", "Churn"], y=["Stay", "Churn"],
                        labels=dict(x="Predicted", y="Actual"), title="Confusion matrix")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     title="Top 10 features (permutation importance, AUC drop)",
                     color_discrete_sequence=[AMBER])
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
    with c4:
        ths = np.linspace(0.05, 0.95, 37)
        pr = [precision_score(y_true, y_prob >= t, zero_division=0) for t in ths]
        rc = [recall_score(y_true, y_prob >= t) for t in ths]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ths, y=pr, name="Precision", line=dict(color=AMBER, width=3)))
        fig.add_trace(go.Scatter(x=ths, y=rc, name="Recall", line=dict(color=RED, width=3)))
        fig.update_layout(title="Precision vs recall by threshold", xaxis_title="Decision threshold")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        st.caption("Lower the threshold to catch more churners (higher recall) "
                   "at the cost of more false alarms.")


def page_live():
    st.title("🚀 Live App")
    t1, t2, t3, t4 = st.tabs(["🏠 Dashboard", "🔮 Predict", "📂 Batch", "📊 Analytics"])
    with t1: tab_dashboard()
    with t2: tab_predict()
    with t3: tab_batch()
    with t4: tab_analytics()


# --------------------------------------------------------------------------
# PAGE 5 - LEARNINGS
# --------------------------------------------------------------------------
def page_learnings():
    st.title("💡 Learnings")
    cols = st.columns(len(LEARNINGS))
    for col, (title, text) in zip(cols, LEARNINGS):
        with col:
            card(title, text)
    st.markdown("&nbsp;")
    st.markdown("**Tech stack:** Python · pandas · scikit-learn · XGBoost · Plotly · Streamlit · Git/GitHub")
    st.caption("Built as part of The IoT Academy - Applied Data Science & ML Internship.")


# --------------------------------------------------------------------------
# NAVIGATION
# --------------------------------------------------------------------------
PAGES = {
    "🎯 Problem Statement": page_problem,
    "📊 EDA Highlights": page_eda,
    "🤖 Model Approach": page_model,
    "🚀 Live App": page_live,
    "💡 Learnings": page_learnings,
}

with st.sidebar:
    st.markdown("## 📡 ChurnSense")
    st.caption("Telco customer churn prediction")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Data: " + ("✅ loaded" if df is not None else "❌ not found"))
    st.caption("Model: " + ("✅ loaded" if model is not None else "❌ not found"))

PAGES[choice]()