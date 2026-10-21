"""Interactive customer segmentation demo backed by the project's K-Means model."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).parent / "data" / "clustering" / "Online Retail.xlsx"
FEATURES = ["recency", "min_amount", "max_amount", "total_amount", "frequency"]


def make_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Match the feature engineering used in notebooks/Clustering.ipynb."""
    data = transactions.copy()
    data = data.dropna(subset=["CustomerID"])
    data = data[~data["InvoiceNo"].astype(str).str.startswith("C")]
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce").fillna(0)
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce").fillna(0.0)
    data["Amount"] = data["Quantity"] * data["UnitPrice"]
    reference_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)

    customers = data.groupby("CustomerID").agg(
        recency=("InvoiceDate", lambda dates: (reference_date - dates.max()).days),
        min_amount=("Amount", "min"),
        max_amount=("Amount", "max"),
        total_amount=("Amount", "sum"),
        frequency=("InvoiceNo", "nunique"),
    )

    # Same upper-IQR filter on maximum order value as the notebook.
    q1, q3 = customers["max_amount"].quantile([0.25, 0.75])
    return customers[customers["max_amount"] <= q3 + 1.5 * (q3 - q1)]


def segment_names(profile: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """Give each numeric K-Means cluster a readable, data-driven description."""
    remaining = set(profile.index)
    names: dict[int, tuple[str, str]] = {}

    champion = (profile["total_amount"] * profile["frequency"] / (profile["recency"] + 1)).idxmax()
    names[int(champion)] = ("Loyal high-value customer", "Recent, frequent shoppers with strong total spend.")
    remaining.remove(champion)

    lapsed = profile.loc[list(remaining), "recency"].idxmax()
    names[int(lapsed)] = ("At-risk or lapsed customer", "Customers whose most recent purchase was comparatively long ago.")
    remaining.remove(lapsed)

    high_ticket = profile.loc[list(remaining), "min_amount"].idxmax()
    names[int(high_ticket)] = ("Occasional high-ticket shopper", "Less frequent customers whose orders tend to start at a higher value.")
    remaining.remove(high_ticket)

    for cluster_id in sorted(remaining, key=lambda item: profile.loc[item, "total_amount"], reverse=True):
        if not any(label == "Valuable regular customer" for label, _ in names.values()):
            names[int(cluster_id)] = ("Valuable regular customer", "Active customers with solid purchase frequency and spend.")
        else:
            names[int(cluster_id)] = ("Developing or casual customer", "Customers with lower engagement and spend than the core groups.")
    return names


@st.cache_resource(show_spinner="Training the 5-cluster K-Means model…")
def train_model():
    transactions = pd.read_excel(DATA_PATH)
    customers = make_customer_features(transactions)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(customers[FEATURES])
    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    customers = customers.copy()
    customers["cluster"] = model.fit_predict(scaled)
    profile = customers.groupby("cluster")[FEATURES].mean()
    return scaler, model, customers, profile, segment_names(profile)


st.set_page_config(page_title="Customer Cluster Finder", page_icon="🛍️", layout="wide")
st.title("Customer Cluster Finder")
st.caption("A 5-cluster K-Means segmentation demo trained on the Online Retail dataset.")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: `{DATA_PATH.relative_to(Path(__file__).parent)}`")
    st.stop()

scaler, model, customers, profile, names = train_model()

with st.sidebar:
    st.header("Enter customer details")
    st.caption("Amounts use the dataset's currency. Recency is days since the customer's last purchase.")
    recency = st.number_input("Days since last purchase", min_value=0, value=45, step=1)
    frequency = st.number_input("Number of purchases", min_value=1, value=3, step=1)
    total_amount = st.number_input("Total amount spent", min_value=0.01, value=800.0, step=10.0)
    min_amount = st.number_input("Smallest order value", min_value=0.0, value=5.0, step=1.0)
    max_amount = st.number_input("Largest order value", min_value=0.0, value=100.0, step=5.0)
    classify = st.button("Find customer cluster", type="primary", use_container_width=True)

st.subheader("How the model segments customers")
st.write("K-Means groups customers by similarity; the labels below are interpretations of the resulting groups, not rules hard-coded into the model.")

summary = profile.copy()
summary.insert(0, "Customer type", [names[int(cluster)][0] for cluster in summary.index])
summary.index.name = "Cluster"
summary = summary.rename(columns={
    "recency": "Avg. days since purchase", "frequency": "Avg. purchases",
    "total_amount": "Avg. total spend", "min_amount": "Avg. smallest order", "max_amount": "Avg. largest order",
}).round(1)
st.dataframe(summary, use_container_width=True)

if classify:
    if max_amount < min_amount:
        st.error("Largest order value must be greater than or equal to smallest order value.")
    elif total_amount < max_amount:
        st.error("Total amount spent must be greater than or equal to largest order value.")
    else:
        entry = pd.DataFrame([[recency, min_amount, max_amount, total_amount, frequency]], columns=FEATURES)
        cluster = int(model.predict(scaler.transform(entry))[0])
        label, explanation = names[cluster]
        average = profile.loc[cluster]
        st.success(f"This customer belongs to Cluster {cluster}: {label}.")
        st.write(explanation)
        a, b, c = st.columns(3)
        a.metric("Days since last purchase", f"{recency:,}", f"cluster avg. {average['recency']:.0f}")
        b.metric("Purchases", f"{frequency:,}", f"cluster avg. {average['frequency']:.1f}")
        c.metric("Total spend", f"{total_amount:,.2f}", f"cluster avg. {average['total_amount']:,.2f}")

st.divider()
st.caption(f"Model trained on {len(customers):,} eligible customers using recency, frequency, total spend, and order-value features.")
