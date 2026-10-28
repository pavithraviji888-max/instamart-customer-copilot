import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import resample
import joblib

def build_lapse_model(segments):
    from pathlib import Path
    Path("outputs").mkdir(parents=True, exist_ok=True)
    
    orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
    cutoff = pd.Timestamp("2025-09-01")
    pre = orders[(orders.order_ts < cutoff) & orders.status.isin(["completed","cancelled"])].copy()
    completed_pre = pre[pre.status.eq("completed")]
    cohort_ids = completed_pre.customer_id.unique()
    sept = orders[(orders.order_ts >= cutoff) & (orders.order_ts < pd.Timestamp("2025-10-01")) & orders.status.eq("completed")]
    active_ids = set(sept.customer_id)
    labels = pd.DataFrame({"customer_id": cohort_ids})
    labels["lapsed"] = (~labels.customer_id.isin(active_ids)).astype(int)

    agg = completed_pre.groupby("customer_id").agg(
        last_order_ts=("order_ts","max"), frequency=("order_id","nunique"),
        monetary=("order_amount","sum"), avg_discount=("discount_pct","mean"),
        avg_delivery=("delivery_minutes","mean")
    ).reset_index()
    agg["recency"] = (cutoff - agg.last_order_ts).dt.days
    total = pre.groupby("customer_id").size().rename("all_final_orders")
    cancels = pre[pre.status.eq("cancelled")].groupby("customer_id").size().rename("cancelled_orders")
    agg = agg.merge(total, on="customer_id", how="left").merge(cancels, on="customer_id", how="left")
    agg["cancelled_orders"] = agg.cancelled_orders.fillna(0)
    agg["cancellation_rate"] = agg.cancelled_orders / agg.all_final_orders
    # Favourite category based only on pre-cutoff completed orders.
    cat_counts = completed_pre.merge(pd.read_csv("data/clean/products.csv")[["product_id","category"]], on="product_id")
    fav = cat_counts.groupby("customer_id")["category"].agg(lambda s: s.value_counts().index[0]).rename("favourite_category")
    agg = agg.merge(fav, on="customer_id", how="left")
    data = labels.merge(agg, on="customer_id", how="left").merge(customers, on="customer_id", how="left")
    data["tenure_days"] = (cutoff - data.signup_date).dt.days
    data = data.drop(columns=["last_order_ts","all_final_orders","cancelled_orders","signup_date"])
    data = data.replace([np.inf,-np.inf], np.nan)
    numeric = ["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","age","tenure_days"]
    categorical = ["favourite_category","city","gender"]
    X = data[numeric + categorical]
    y = data["lapsed"].astype(int)
    assert X.isna().sum().sum() == 0, "Unexpected missing feature values"
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, data.customer_id, test_size=0.2, stratify=y, random_state=42
    )
    num_pipe = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat_pipe = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(drop="first",handle_unknown="ignore"))])
    prep = ColumnTransformer([("num",num_pipe,numeric),("cat",cat_pipe,categorical)])
    lr_pipe = Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1000))])
    lr_grid = GridSearchCV(lr_pipe, {"model__C":[0.01,0.1,1,10,100]}, cv=5, scoring="f1", n_jobs=-1)
    lr_grid.fit(X_train,y_train)
    rf_pipe = Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    rf_grid = GridSearchCV(rf_pipe, {"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]}, cv=5, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train,y_train)
    def report(name, model):
        pred = model.predict(X_test)
        print(name, "best params:", model.best_params_, "CV F1:", model.best_score_)
        print("confusion matrix:\n", confusion_matrix(y_test,pred))
        print("precision/recall/F1:", precision_score(y_test,pred,zero_division=0), recall_score(y_test,pred,zero_division=0), f1_score(y_test,pred,zero_division=0))
        return pred
    lr_pred = report("LogisticRegression",lr_grid)
    rf_pred = report("RandomForest",rf_grid)
    selected = rf_grid if f1_score(y_test,rf_pred,zero_division=0) >= f1_score(y_test,lr_pred,zero_division=0) else lr_grid
    joblib.dump(selected, "outputs/lapse_model.joblib")
    loaded = joblib.load("outputs/lapse_model.joblib")
    assert np.array_equal(selected.predict(X_test), loaded.predict(X_test))
    probs = selected.predict_proba(X_test)[:,1]
    seg_lookup = segments[["customer_id","is_premium"]]
    ret = pd.DataFrame({"customer_id":id_test.to_numpy(),"lapse_probability":probs,"actually_lapsed":y_test.to_numpy()})
    ret = ret.merge(seg_lookup,on="customer_id",how="left")
    ret = ret[ret.is_premium.fillna(False)].sort_values("lapse_probability",ascending=False).head(10)
    ret[["customer_id","lapse_probability","actually_lapsed"]].to_csv("outputs/retention_list.csv",index=False)
    print("Retention-list true lapses:", int(ret.actually_lapsed.sum()))
    return selected
