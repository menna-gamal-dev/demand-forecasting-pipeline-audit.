# Case 2: Demand Forecasting Pipeline Health Audit

## Executive Summary
This audit evaluates an automated nightly demand forecasting pipeline for a retail supply chain. The audit quantifies model error rates, pinpoints execution failures, and identifies severe performance gaps during promotional days.

---

## Required Case 2 Answers (Q13 - Q17)

1. **Pipeline Run Status (Q13):**
   - **Failed/Delayed Runs:** Excluded from prediction accuracy scoring because failed runs produce no valid forecast values. However, they are tracked to measure operational pipeline reliability.
2. **Overall Accuracy (Q14):**
   - Evaluated using Mean Absolute Percentage Error (MAPE) across successful runs.
3. **Promotion Impact & Missing Inputs (Q15):**
   - **Findings:** Forecast MAPE is significantly higher on promotion days (`is_promotion = 1`) compared to non-promotion days.
   - **Root Cause:** The model lacks explicit promo-lift features (e.g., promotional discount percentages, campaign intensity), causing systemic under-prediction during sales surges.
4. **Worst Performing Category (Q16):**
   - Identified product category with the highest error rate during promotions to highlight high inventory-risk items.
5. **Recommendations & Action Plan (Q17):**
   - **Short-term Supply Chain Action:** Apply a temporary safety-stock buffer multiplier (e.g., 1.2x - 1.5x) for high-risk categories on scheduled promotion days.
   - **Model Upgrades:** Inject discount rates, promotional flags, and category-level promotion response multipliers directly into the feature generation pipeline.

---

## Feature Engineering Justification
- `promo_x_category`: Captures category-specific demand elasticity during promotional campaigns.
- `is_pipeline_issue`: Flags delayed/failed pipeline runs to alert downstream inventory replenishment modules.

---

## Production-Readiness & Limitations
- **Silently Breaking Risks:** Unexpected string formats in `date` or changes in category schema will silently break downstream aggregations.
- **Assumptions:** Missing sales records during non-failed runs are assumed to be zero demand rather than missing data pipeline loss.
-