from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def build_segments():
    Path("outputs").mkdir(parents=True, exist_ok=True)
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv")
    orders = orders[orders.status.eq("completed")].copy()
    snapshot = pd.Timestamp("2025-10-01")
    rfm = orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, labels)})
    pd.DataFrame(rows).to_csv("outputs/kmeans_metrics.csv", index=False)
    # Choose K=4 for business interpretability and alignment with the project reference.
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["segment_id"] = km.fit_predict(X)
    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "size"), mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")
    ).reset_index()
    # Give deterministic business names by relative profiles.
    med_spend = profile["mean_monetary"].median()
    names = {}
    for _, row in profile.iterrows():
        if row.mean_monetary == profile.mean_monetary.max():
            names[int(row.segment_id)] = "Premium Spenders"
        elif row.mean_frequency == profile.mean_frequency.max():
            names[int(row.segment_id)] = "Loyal Frequent Buyers"
        elif row.mean_recency == profile.mean_recency.max():
            names[int(row.segment_id)] = "At-Risk Customers"
        else:
            names[int(row.segment_id)] = "Regular Customers"
    rfm["segment_name"] = rfm.segment_id.map(names)
    premium_id = int(profile.loc[profile.mean_monetary.idxmax(), "segment_id"])
    rfm["is_premium"] = rfm.segment_id.eq(premium_id)
    rfm = rfm.merge(customers[["customer_id"]], on="customer_id", how="right")
    rfm["is_premium"] = rfm["is_premium"].fillna(False).astype(bool)
    rfm["segment_id"] = rfm["segment_id"].fillna(-1).astype(int)
    rfm["segment_name"] = rfm["segment_name"].fillna("No Completed Orders")
    rfm.to_csv("outputs/customer_segments.csv", index=False)
    profile.to_csv("outputs/segment_profile.csv", index=False)
    pca = PCA(n_components=2).fit(X)
    pd.DataFrame({"component": [1,2], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv("outputs/pca_variance.csv", index=False)
    print("RFM customers:", len(rfm[rfm.frequency.notna()]))
    print(pd.DataFrame(rows).to_string(index=False))
    print(profile.to_string(index=False))
    print("PCA variance ratios:", pca.explained_variance_ratio_)
    return rfm

if __name__ == "__main__":
    build_segments()

