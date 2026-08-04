import pandas as pd
import numpy as np
from src.exceptions import DataQualityError

class DemandPipelineAudit:
    """
    Core Domain Class representing a single Demand Forecasting Pipeline run audit.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None

    def load_and_validate(self, max_invalid_threshold: float = 0.3):
        """Rule 1: Data loading & validation pattern with custom DataQualityError."""
        self.df = pd.read_csv(self.data_path)
        
        # Checking invalid/corrupted records in successful runs
        invalid_rows = self.df[
            (self.df['pipeline_run_status'] == 'Success') & 
            (self.df['actual_units_sold'].isna() | self.df['predicted_demand_units'].isna())
        ]
        invalid_fraction = len(invalid_rows) / len(self.df) if len(self.df) > 0 else 0
        
        if invalid_fraction > max_invalid_threshold:
            raise DataQualityError(
                f"Data quality issue: {invalid_fraction:.2%} invalid rows exceeds threshold {max_invalid_threshold:.2%}"
            )
        return self.df

    def engineer_features(self):
        """Rule 4: Create 2 new domain features for future ML models."""
        df = self.df.copy()
        # Feature 1: Promo x Category interaction string
        df['promo_x_category'] = df['is_promotion'].astype(str) + "_" + df['product_category'].astype(str)
        # Feature 2: Unreliability flag for failed or delayed runs
        df['is_pipeline_issue'] = df['pipeline_run_status'].isin(['Failed', 'Delayed']).astype(int)
        
        self.df = df
        return df

    def run_full_analysis(self):
        """Rule 2: Answer all 5 required business questions explicitly."""
        df = self.df
        
        # Q13: Validate pipeline_run_status
        status_counts = df['pipeline_run_status'].value_counts().to_dict()
        failed_or_delayed = status_counts.get('Failed', 0) + status_counts.get('Delayed', 0)
        
        # Drop Failed/Delayed for accuracy evaluation as they lack valid predictions
        success_df = df[df['pipeline_run_status'] == 'Success'].dropna(subset=['actual_units_sold', 'predicted_demand_units']).copy()
        
        # Q14: Overall forecast error & MAPE
        success_df['forecast_error'] = success_df['actual_units_sold'] - success_df['predicted_demand_units']
        success_df['abs_error'] = np.abs(success_df['forecast_error'])
        success_df['abs_pct_error'] = (success_df['abs_error'] / success_df['actual_units_sold'].replace(0, np.nan)) * 100
        overall_mape = success_df['abs_pct_error'].mean()
        
        # Q15: Promo vs Non-Promo MAPE Comparison
        promo_mape = success_df[success_df['is_promotion'] == 1]['abs_pct_error'].mean()
        non_promo_mape = success_df[success_df['is_promotion'] == 0]['abs_pct_error'].mean()
        
        # Q16: Worst product_category during promotions
        promo_df = success_df[success_df['is_promotion'] == 1]
        cat_mape = promo_df.groupby('product_category')['abs_pct_error'].mean()
        worst_category = cat_mape.idxmax() if not cat_mape.empty else "N/A"
        worst_category_mape = cat_mape.max() if not cat_mape.empty else 0.0

        return {
            "status_counts": status_counts,
            "failed_or_delayed": failed_or_delayed,
            "overall_mape": overall_mape,
            "promo_mape": promo_mape,
            "non_promo_mape": non_promo_mape,
            "worst_category": worst_category,
            "worst_category_mape": worst_category_mape
        }