from src.pipeline_audit import DemandPipelineAudit

def main():
    print("==================================================")
    print(" Demand Forecasting Pipeline Health Audit")
    print("==================================================\n")
    
    auditor = DemandPipelineAudit(data_path="data/demand_forecast_pipeline.csv")
    auditor.load_and_validate()
    auditor.engineer_features()
    
    results = auditor.run_full_analysis()
    
    print(f"1. Run Statuses Frequencies: {results['status_counts']}")
    print(f"   -> Failed or Delayed Runs: {results['failed_or_delayed']}")
    print(f"2. Overall Mean Absolute Percentage Error (MAPE): {results['overall_mape']:.2f}%")
    print(f"3. Promo MAPE: {results['promo_mape']:.2f}% vs Non-Promo MAPE: {results['non_promo_mape']:.2f}%")
    print(f"4. Worst Category During Promo: {results['worst_category']} (MAPE: {results['worst_category_mape']:.2f}%)\n")

if __name__ == "__main__":
    main()