import sqlite3
import pandas as pd

def validate():
    customers = pd.read_csv("data/clean/customers.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    assert len(customers) == 1200
    assert customers["city"].nunique() == 6
    assert customers["age"].notna().all()
    assert customers["age"].between(18, 60).all()
    assert set(customers["gender"].unique()) <= {"male", "female", "unknown"}
    assert len(orders) == 12309
    assert orders["order_id"].nunique() == 12309
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].notna().all()
    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    assert orders.loc[completed, "delivery_minutes"].notna().all()
    assert orders.loc[cancelled, "delivery_minutes"].isna().all()
    assert abs(orders.loc[completed, "order_amount"].sum() - 2998668.80) < 0.02
    with sqlite3.connect("data/instamart.db") as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"customers", "products", "orders"} <= tables
    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate()
