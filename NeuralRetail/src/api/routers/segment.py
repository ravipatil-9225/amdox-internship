from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from src.api.security import get_current_user
from src.config.settings import settings
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

router = APIRouter()

class SegmentRequest(BaseModel):
    customer_id: Optional[str] = None

class CustomerSegment(BaseModel):
    customer_id: str
    segment: str
    segment_id: int
    is_outlier: bool
    gmm_segment_id: int
    rfm_recency: float
    rfm_frequency: float
    rfm_monetary: float
    lifetime_value: float

class SegmentResponse(BaseModel):
    total_customers: int
    num_segments: int
    segments: List[CustomerSegment]
    segment_summary: Dict[str, Dict[str, float]]

# Cache
_segmentation_result = None

def run_segmentation():
    global _segmentation_result
    if _segmentation_result is not None:
        return _segmentation_result

    try:
        customers = pd.read_parquet(settings.data_dir / "customers.parquet")
        transactions = pd.read_parquet(settings.data_dir / "transactions.parquet")

        latest_date = transactions['timestamp'].max()
        rfm = transactions.groupby('customer_id').agg({
            'timestamp': lambda x: (latest_date - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).reset_index()
        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        # Standardize for clustering (downsample if too large for real-time API performance)
        if len(rfm) > 2000:
            rfm_sample = rfm.sample(n=2000, random_state=42).copy()
        else:
            rfm_sample = rfm.copy()

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_sample[['recency', 'frequency', 'monetary']])

        # 1. K-Means with 6 segments
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        rfm_sample['segment_id'] = kmeans.fit_predict(rfm_scaled)

        # 2. DBSCAN for outlier detection
        dbscan = DBSCAN(eps=0.8, min_samples=5)
        rfm_sample['dbscan_cluster'] = dbscan.fit_predict(rfm_scaled)
        rfm_sample['is_outlier'] = rfm_sample['dbscan_cluster'] == -1

        # 3. Gaussian Mixture Model for probabilistic assignments
        gmm = GaussianMixture(n_components=6, random_state=42)
        rfm_sample['gmm_segment_id'] = gmm.fit_predict(rfm_scaled)

        # Predict back on full dataset for consistent output
        full_scaled = scaler.transform(rfm[['recency', 'frequency', 'monetary']])
        rfm['segment_id'] = kmeans.predict(full_scaled)
        # DBSCAN doesn't have predict, so we use KNN or just map -1 if distance is too far. For speed, we approximate:
        rfm['is_outlier'] = False  # Default to false for non-sampled points to save time
        rfm.loc[rfm_sample[rfm_sample['is_outlier']].index, 'is_outlier'] = True
        rfm['gmm_segment_id'] = gmm.predict(full_scaled)

        # Assign segment labels based on cluster centroids
        segment_labels = {}
        centroids = kmeans.cluster_centers_
        for i in range(6):
            rec, freq, mon = centroids[i]
            if mon > 0.5 and freq > 0.5:
                segment_labels[i] = "VIP Champions"
            elif freq > 0.3 and rec < 0:
                segment_labels[i] = "Loyal Customers"
            elif rec < -0.5:
                segment_labels[i] = "Recent Buyers"
            elif rec > 0.5 and freq < 0:
                segment_labels[i] = "At Risk"
            elif rec > 1.0:
                segment_labels[i] = "Lost Customers"
            else:
                segment_labels[i] = "Potential Loyalists"

        rfm['segment'] = rfm['segment_id'].map(segment_labels)
        rfm['lifetime_value'] = rfm['monetary'] * (rfm['frequency'] / (rfm['recency'] + 1))

        _segmentation_result = rfm
        return rfm
    except Exception as e:
        print(f"Segmentation error: {e}")
        return None

@router.post("/score", response_model=SegmentResponse)
async def segment_customer(request: SegmentRequest, current_user = Depends(get_current_user)):
    """
    Run K-Means segmentation on all customers and return segment assignments.
    Optionally filter by customer_id.
    """
    rfm = run_segmentation()
    if rfm is None:
        raise HTTPException(status_code=500, detail="Segmentation failed.")

    if request.customer_id:
        filtered = rfm[rfm['customer_id'] == request.customer_id]
        if filtered.empty:
            filtered = rfm.head(10)
    else:
        filtered = rfm.head(50)

    segments_list = []
    for _, row in filtered.iterrows():
        segments_list.append(CustomerSegment(
            customer_id=row['customer_id'],
            segment=row['segment'],
            segment_id=int(row['segment_id']),
            is_outlier=bool(row['is_outlier']),
            gmm_segment_id=int(row['gmm_segment_id']),
            rfm_recency=round(float(row['recency']), 2),
            rfm_frequency=round(float(row['frequency']), 2),
            rfm_monetary=round(float(row['monetary']), 2),
            lifetime_value=round(float(row['lifetime_value']), 2)
        ))

    # Summary stats per segment
    summary = {}
    for seg_name in rfm['segment'].unique():
        seg_data = rfm[rfm['segment'] == seg_name]
        summary[seg_name] = {
            "count": float(len(seg_data)),
            "avg_recency": round(float(seg_data['recency'].mean()), 2),
            "avg_frequency": round(float(seg_data['frequency'].mean()), 2),
            "avg_monetary": round(float(seg_data['monetary'].mean()), 2),
            "total_revenue": round(float(seg_data['monetary'].sum()), 2)
        }

    return SegmentResponse(
        total_customers=len(rfm),
        num_segments=rfm['segment'].nunique(),
        segments=segments_list,
        segment_summary=summary
    )
