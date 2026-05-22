import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# Ensure output directories exist
os.makedirs("data/bronze", exist_ok=True)

# Configuration for Data Generation
NUM_CUSTOMERS = 5000
NUM_PRODUCTS = 200
NUM_TRANSACTIONS = 50000
DAYS_OF_HISTORY = 365

print("Starting enterprise synthetic data generation...")

# 1. Generate Products (SKUs)
print("Generating Products...")
categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Toys"]
products = pd.DataFrame({
    "sku_id": [f"SKU-{1000 + i}" for i in range(NUM_PRODUCTS)],
    "category": np.random.choice(categories, NUM_PRODUCTS),
    "base_price": np.round(np.random.uniform(10.0, 500.0, NUM_PRODUCTS), 2),
    "cost_price": np.round(np.random.uniform(5.0, 200.0, NUM_PRODUCTS), 2)
})
# Ensure cost is always lower than base price
products['cost_price'] = np.where(products['cost_price'] >= products['base_price'], products['base_price'] * 0.6, products['cost_price'])

# 2. Generate Customers
print("Generating Customers...")
segments = ["New", "Regular", "VIP", "At Risk"]
customers = pd.DataFrame({
    "customer_id": [f"CUST-{10000 + i}" for i in range(NUM_CUSTOMERS)],
    "age": np.random.randint(18, 75, NUM_CUSTOMERS),
    "gender": np.random.choice(["M", "F", "Other"], NUM_CUSTOMERS, p=[0.48, 0.48, 0.04]),
    "signup_date": [datetime.now() - timedelta(days=random.randint(10, DAYS_OF_HISTORY)) for _ in range(NUM_CUSTOMERS)],
    "segment": np.random.choice(segments, NUM_CUSTOMERS, p=[0.2, 0.5, 0.1, 0.2])
})

# 3. Generate Transactions (Sales History)
print("Generating Transactions...")
end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS_OF_HISTORY)

# Create random dates within the last year
transaction_dates = [start_date + timedelta(days=random.randint(0, DAYS_OF_HISTORY), 
                                            hours=random.randint(0, 23), 
                                            minutes=random.randint(0, 59)) 
                     for _ in range(NUM_TRANSACTIONS)]

transactions = pd.DataFrame({
    "transaction_id": [f"TXN-{100000 + i}" for i in range(NUM_TRANSACTIONS)],
    "customer_id": np.random.choice(customers["customer_id"], NUM_TRANSACTIONS),
    "sku_id": np.random.choice(products["sku_id"], NUM_TRANSACTIONS),
    "timestamp": transaction_dates,
    "quantity": np.random.randint(1, 6, NUM_TRANSACTIONS),
    "discount_applied": np.random.choice([0.0, 0.1, 0.2], NUM_TRANSACTIONS, p=[0.7, 0.2, 0.1])
})

# Merge with products to calculate final revenue
transactions = transactions.merge(products[['sku_id', 'base_price']], on='sku_id', how='left')
transactions['total_amount'] = np.round((transactions['quantity'] * transactions['base_price']) * (1 - transactions['discount_applied']), 2)
transactions = transactions.drop(columns=['base_price'])

# 4. Generate Inventory Data
print("Generating Inventory Data...")
inventory = pd.DataFrame({
    "sku_id": products["sku_id"],
    "current_stock": np.random.randint(0, 500, NUM_PRODUCTS),
    "reorder_point": np.random.randint(20, 100, NUM_PRODUCTS),
    "supplier_lead_time_days": np.random.randint(2, 14, NUM_PRODUCTS)
})

# Save to Bronze Layer (Parquet format for performance and compression)
print("Saving datasets to data/bronze/...")
products.to_parquet("data/bronze/products.parquet", index=False)
customers.to_parquet("data/bronze/customers.parquet", index=False)
transactions.to_parquet("data/bronze/transactions.parquet", index=False)
inventory.to_parquet("data/bronze/inventory.parquet", index=False)

print("✅ Data generation complete! Datasets are ready for ML pipelines.")
