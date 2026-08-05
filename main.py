import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.pipeline_audit import DemandForecastAudit
 
def plot_audit_charts(audit_obj: DemandForecastAudit):
    """Generates and displays visual analytics charts for the audit."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # 1. Pipeline Status Breakdown Pie Chart
    status_counts = audit_obj.get_run_status_summary()
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    axes[0].pie(
        status_counts, 
        labels=status_counts.index, 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=140, 
        explode=(0, 0.1, 0.1)
    )
    axes[0].set_title('Pipeline Run Status Breakdown', fontsize=12, fontweight='bold')
 
    # 2. Promo vs Non-Promo MAPE Comparison Bar Chart
    clean_df = audit_obj.clean_df
    promo_comp = clean_df.groupby('is_promotion')['abs_perc_error'].mean().reset_index()
    promo_comp['is_promotion'] = promo_comp['is_promotion'].map({0: 'Non-Promo Days', 1: 'Promo Days'})
 
    sns.barplot(data=promo_comp, x='is_promotion', y='abs_perc_error', ax=axes[1], palette=['#3498db', '#e74c3c'])
    axes[1].set_title('Forecast Error (MAPE %) by Promotion Status', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Mean Absolute Percentage Error (%)')
    axes[1].set_xlabel('')
 
    for p in axes[1].patches:
        axes[1].annotate(
            f'{p.get_height():.1f}%', 
            (p.get_x() + p.get_width() / 2., p.get_height() / 2),
            ha='center', va='center', color='white', fontweight='bold', fontsize=11
        )
 
    plt.tight_layout()
    plt.show()
 
def main():
    data_path = os.path.join('data', 'demand_forecast_pipeline.csv')
    print("=" * 60)
    print("      DEMAND FORECASTING PIPELINE HEALTH - AUDIT REPORT      ")
    print("=" * 60)
 
    # Initialize Audit Object
    audit = DemandForecastAudit(filepath=data_path)
 
    # Q13: Validate pipeline run status
    print("\n[Q13] Pipeline Run Status Breakdown:")
    status_summary = audit.get_run_status_summary()
    print(status_summary.to_string())
 
    # Q14: Overall forecast error and MAPE
    overall_metrics = audit.compute_overall_metrics()
    print("\n[Q14] Forecast Accuracy Metrics (Successful Runs):")
    print(f"    - Overall Mean Forecast Error (Actual - Predicted): {overall_metrics['mean_forecast_error']:.2f} units")
    print(f"    - Mean Absolute Percentage Error (MAPE): {overall_metrics['mape_percent']:.2f}%")
 
    # Q15: Promo vs Non-Promo Comparison
    promo_metrics = audit.compare_promotion_impact()
    print("\n[Q15] Promotion Impact on Accuracy:")
    print(f"    - Promotion Days MAPE:     {promo_metrics['promo_mape']:.2f}%")
    print(f"    - Non-Promotion Days MAPE: {promo_metrics['non_promo_mape']:.2f}%")
    print(f"    - Performance Gap:         +{promo_metrics['gap']:.2f}% higher error during promotions!")
 
    # Q16: Worst Product Category on Promo Days
    worst_cat, worst_mape, all_cat_mape = audit.worst_category_during_promotions()
    print("\n[Q16] Category Accuracy During Promotion Days:")
    for cat, val in all_cat_mape.items():
        print(f"    - {cat}: {val:.2f}% MAPE")
    print(f"\n--> Category with Worst Forecast Accuracy: {worst_cat} ({worst_mape:.2f}% MAPE)")
 
    print("\n" + "=" * 60)
    print("Displaying Analytical Visualizations...")
    print("=" * 60)
    
    # Display Plots
    plot_audit_charts(audit)
 
if __name__ == "__main__":
    main()
plt.show
 