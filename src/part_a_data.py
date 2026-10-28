from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

def build_part_a():
    customers = pd.read_csv(RAW / "customers.csv", keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv")
    orders = pd.read_csv(RAW / "orders.csv", keep_default_na=False)

    # Customers
    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    customers["gender"] = customers["gender"].astype(str).str.strip().str.lower()
    customers["gender"] = customers["gender"].map({
        "m": "male", "male": "male", "f": "female", "female": "female",
        "": "unknown", "nan": "unknown"
    }).fillna("unknown")
    customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
    customers.loc[~customers["age"].between(10, 100), "age"] = np.nan
    customers["age"] = customers["age"].fillna(
        customers.groupby("city")["age"].transform("median")
    )
    customers["age"] = customers["age"].fillna(customers["age"].median()).round().astype(int)
    customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors="coerce")
    customers["signup_date"] = customers["signup_date"].dt.strftime("%Y-%m-%d")

    # Products
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")

    # Orders, in required order
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
    orders = orders.drop_duplicates()
    orders = orders.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last")
    for c in ["quantity", "discount_pct", "order_amount", "delivery_minutes"]:
        orders[c] = pd.to_numeric(orders[c], errors="coerce")
    orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left", validate="many_to_one")
    missing_amount = orders["order_amount"].isna()
    orders.loc[missing_amount, "order_amount"] = (
        orders.loc[missing_amount, "price_inr"] * orders.loc[missing_amount, "quantity"] *
        (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    ).round(2)
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    completed = orders["status"].eq("completed")
    category_median = orders.loc[completed].groupby("category")["delivery_minutes"].transform("median")
    # Map medians by category so all completed rows can be filled consistently.
    medians = orders.loc[completed].groupby("category")["delivery_minutes"].median()
    fill_mask = completed & orders["delivery_minutes"].isna()
    orders.loc[fill_mask, "delivery_minutes"] = orders.loc[fill_mask, "category"].map(medians)
    orders.loc[~completed, "delivery_minutes"] = np.nan
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    orders = orders.drop(columns=["category", "price_inr"])
    orders["quantity"] = orders["quantity"].astype(int)
    orders["discount_pct"] = orders["discount_pct"].astype(int)
    orders["order_amount"] = orders["order_amount"].round(2)

    customers.to_csv(CLEAN / "customers.csv", index=False)
    products.to_csv(CLEAN / "products.csv", index=False)
    orders.to_csv(CLEAN / "orders.csv", index=False)

    db_path = ROOT / "data" / "instamart.db"
    with sqlite3.connect(db_path) as con:
        customers.to_sql("customers", con, if_exists="replace", index=False)
        products.to_sql("products", con, if_exists="replace", index=False)
        orders.to_sql("orders", con, if_exists="replace", index=False)

    # Persist seven SQL queries
    sql_dir = ROOT / "sql"
    sql_dir.mkdir(exist_ok=True)
    (sql_dir / "queries.sql").write_text(SQL_QUERIES, encoding="utf-8")
    print("Part A created clean CSVs and data/instamart.db")
    return customers, products, orders

SQL_QUERIES = """-- Q1 WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS qualifying_spend
FROM orders
WHERE status='completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2 revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c USING(customer_id)
JOIN products p USING(product_id)
WHERE o.status='completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3 products priced above average catalogue price
SELECT product_id, product_name, price_inr
FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4 customers with no completed order
SELECT c.customer_id
FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL;

-- Q5 CTE totals and average customer spend
WITH spend AS (
  SELECT customer_id, SUM(order_amount) AS total_spend
  FROM orders WHERE status='completed' GROUP BY customer_id
), avg_spend AS (
  SELECT AVG(total_spend) AS avg_total FROM spend
)
SELECT spend.customer_id, spend.total_spend
FROM spend CROSS JOIN avg_spend
WHERE spend.total_spend > avg_spend.avg_total;

-- Q6 city-level ranking
WITH spend AS (
 SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
 FROM customers c JOIN orders o USING(customer_id)
 WHERE o.status='completed'
 GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend,2) AS total_spend,
 RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_num
FROM spend;

-- Q7 daily running completed revenue
WITH daily AS (
 SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)
)
SELECT day, ROUND(revenue,2) AS revenue,
 ROUND(SUM(revenue) OVER(ORDER BY day),2) AS cumulative_revenue
FROM daily ORDER BY day;
"""

if __name__ == "__main__":
    build_part_a()