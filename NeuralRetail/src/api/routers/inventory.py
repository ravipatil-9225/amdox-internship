from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.api.security import get_current_user
import pandas as pd
import numpy as np
import math

router = APIRouter()

class InventoryRequest(BaseModel):
    sku_id: str
    store_id: str

class InventoryResponse(BaseModel):
    sku_id: str
    current_stock: int
    recommended_reorder_qty: int
    safety_stock: int
    stockout_risk: float
    days_of_supply: float
    abc_class: str
    dead_stock: bool

@router.post("/reorder", response_model=InventoryResponse)
async def inventory_reorder(request: InventoryRequest, current_user = Depends(get_current_user)):
    """
    EOQ-based inventory optimization with ABC classification and dead-stock detection.
    """
    try:
        inventory = pd.read_parquet("data/bronze/inventory.parquet")
        transactions = pd.read_parquet("data/bronze/transactions.parquet")
        products = pd.read_parquet("data/bronze/products.parquet")

        # Current stock
        inv_row = inventory[inventory['sku_id'] == request.sku_id]
        current_stock = int(inv_row['current_stock'].values[0]) if not inv_row.empty else 0

        # Product cost
        prod_row = products[products['sku_id'] == request.sku_id]
        unit_cost = float(prod_row['cost_price'].values[0]) if not prod_row.empty else 10.0

        # Average daily demand from transactions
        sku_txns = transactions[transactions['sku_id'] == request.sku_id]
        if not sku_txns.empty:
            date_range = (sku_txns['timestamp'].max() - sku_txns['timestamp'].min()).days + 1
            total_qty = sku_txns['quantity'].sum()
            avg_daily_demand = total_qty / max(date_range, 1)
            demand_std = sku_txns.groupby(sku_txns['timestamp'].dt.date)['quantity'].sum().std()
        else:
            avg_daily_demand = 5.0
            demand_std = 2.0

        # EOQ Calculation
        annual_demand = avg_daily_demand * 365
        ordering_cost = 50.0  # fixed per order
        holding_cost_rate = 0.20  # 20% of unit cost per year
        holding_cost = unit_cost * holding_cost_rate

        if holding_cost > 0 and annual_demand > 0:
            eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        else:
            eoq = 100

        # Safety stock (z=1.65 for 95% service level, lead time = 7 days)
        lead_time = 7
        z_score = 1.65
        safety = z_score * (demand_std if not np.isnan(demand_std) else 2.0) * math.sqrt(lead_time)

        # Days of supply
        dos = current_stock / max(avg_daily_demand, 0.1)

        # Stockout risk
        reorder_point = avg_daily_demand * lead_time + safety
        stockout_risk = max(0, min(1.0, 1.0 - (current_stock / max(reorder_point, 1))))

        # ABC Classification by revenue contribution
        all_revenue = transactions.groupby('sku_id')['total_amount'].sum().sort_values(ascending=False)
        cumulative = all_revenue.cumsum() / all_revenue.sum()
        if request.sku_id in cumulative.index:
            cum_val = cumulative[request.sku_id]
            abc_class = "A" if cum_val <= 0.7 else ("B" if cum_val <= 0.9 else "C")
        else:
            abc_class = "C"

        # Dead stock: no sales in last 90 days
        if not sku_txns.empty:
            last_sale = sku_txns['timestamp'].max()
            days_since = (pd.Timestamp.now(tz='UTC') - last_sale).days
            dead_stock = days_since > 90
        else:
            dead_stock = True

        return InventoryResponse(
            sku_id=request.sku_id,
            current_stock=current_stock,
            recommended_reorder_qty=int(round(eoq)),
            safety_stock=int(round(safety)),
            stockout_risk=round(stockout_risk, 4),
            days_of_supply=round(dos, 1),
            abc_class=abc_class,
            dead_stock=dead_stock
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inventory optimization failed: {e}")
