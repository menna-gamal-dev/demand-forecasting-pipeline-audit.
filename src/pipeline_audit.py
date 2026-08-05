import pandas as pd
import numpy as np
from src.exceptions import DataQualityError
 
class DemandForecastAudit:
    """
    Class to hold data and audit operations for the Demand Forecasting Pipeline.
    """
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = None
        self.clean_df = None
        self.load_and_validate_data()
 
    def load_and_validate_data(self):
        """Loads data, coerces types, and raises DataQualityError if invalid fraction is too high."""
        try:
            df = pd.read_csv(self.filepath)
        except Exception as e:
            raise DataQualityError(f"Failed to load dataset: {e}")
 
        required_cols = [
            'date', 'store_id', 'store_city', 'product_category',
            'predicted_demand_units', 'actual_units_sold', 'is_promotion', 'pipeline_run_status'
        ]
        
        for col in required_cols:
            if col not in df.columns:
                raise DataQualityError(f"Missing required column: {col}")
 
        # Coerce types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['predicted_demand_units'] = pd.to_numeric(df['predicted_demand_units'], errors='coerce')
        df['actual_units_sold'] = pd.to_numeric(df['actual_units_sold'], errors='coerce')
        df['is_promotion'] = pd.to_numeric(df['is_promotion'], errors='coerce')
 
        # Check critical invalid/missing data in actual_units_sold
        invalid_actuals = df['actual_units_sold'].isna().sum()
        invalid_ratio = invalid_actuals / len(df)
        
        if invalid_ratio > 0.20:  # Threshold > 20%
            raise DataQualityError(f"Data quality error: {invalid_ratio:.2%} of actual sales data is invalid.")
 
        self.raw_df = df.copy()
        
        # Prepare working/successful dataset
        self.clean_df = df[df['pipeline_run_status'] == 'Success'].copy()
        self.clean_df['forecast_error'] = self.clean_df['actual_units_sold'] - self.clean_df['predicted_demand_units']
        self.clean_df['abs_perc_error'] = (
            np.abs(self.clean_df['forecast_error']) / self.clean_df['actual_units_sold']
        ) * 100
 
    def get_run_status_summary(self) -> pd.Series:
        """Returns counts of pipeline run statuses (Success, Failed, Delayed)."""
        return self.raw_df['pipeline_run_status'].value_counts()
 
    def compute_overall_metrics(self) -> dict:
        """Computes mean forecast error and MAPE on successful runs."""
        mean_error = self.clean_df['forecast_error'].mean()
        mape = self.clean_df['abs_perc_error'].mean()
        return {'mean_forecast_error': mean_error, 'mape_percent': mape}
 
    def compare_promotion_impact(self) -> dict:
        """Compares MAPE during promotion vs non-promotion days."""
        promo_mape = self.clean_df[self.clean_df['is_promotion'] == 1]['abs_perc_error'].mean()
        non_promo_mape = self.clean_df[self.clean_df['is_promotion'] == 0]['abs_perc_error'].mean()
        return {
            'promo_mape': promo_mape,
            'non_promo_mape': non_promo_mape,
            'gap': promo_mape - non_promo_mape
        }
 
    def worst_category_during_promotions(self) -> tuple:
        """Identifies product category with worst forecast accuracy (highest MAPE) on promo days."""
        promo_df = self.clean_df[self.clean_df['is_promotion'] == 1]
        category_mape = promo_df.groupby('product_category')['abs_perc_error'].mean()
        worst_category = category_mape.idxmax()
        worst_mape = category_mape.max()
        return worst_category, worst_mape, category_mape
 
    def add_engineered_features(self) -> pd.DataFrame:
        """
        Engineers 2 new features for future ML models:
        1. promo_demand_lift: Measures actual demand variance between promo & non-promo baseline per category.
        2. pipeline_lag_indicator: Flags consecutive system instability / delayed runs.
        """
        df = self.raw_df.copy()
        
        # Feature 1: Historical average sales per store and category (Baseline demand without promo effect)
        baseline = df[df['is_promotion'] == 0].groupby(['store_id', 'product_category'])['actual_units_sold'].transform('mean')
        df['promo_demand_lift'] = (df['actual_units_sold'] - baseline) / baseline
        
        # Feature 2: Is delayed or failed run flag (Pipeline Health Feature)
        df['is_unhealthy_run'] = df['pipeline_run_status'].apply(lambda x: 1 if x in ['Failed', 'Delayed'] else 0)
        
        return df
 

 