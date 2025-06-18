# Feature-Target Relationship Analysis Over Time

## Overview
This report investigates how the relationships between input features and gross power output in the geothermal plant data have changed from 2020 to 2023. The goal is to understand why machine learning models trained on one year systematically underestimate power output in other years, even as the plant's performance degrades.

---

## 1. Correlation Analysis by Year

The correlation between each feature and the target (Gross Power) varies significantly across years:

| Year | Brine Flowrate | Fluid Temp | NCG+Steam Flowrate | Ambient Temp | Reinjection Temp |
|------|----------------|------------|--------------------|--------------|------------------|
| 2020 | 0.30           | 0.63       | 0.36               | -0.68        | -0.43            |
| 2021 | 0.61           | 0.27       | 0.57               | -0.57        | -0.52            |
| 2022 | 0.42           | 0.39       | 0.63               | -0.60        | -0.63            |
| 2023 | 0.69           | 0.41       | 0.61               | -0.64        | -0.58            |

**See:** [Feature-Target Correlations Over Time](feature_correlations_over_time.html)

---

## 2. Linear Regression Coefficients by Year

The standardized regression coefficients (from a linear model) also change year to year, indicating that the mapping from features to output is not stable.

**See:** [Regression Coefficients Over Time](regression_coefficients_over_time.html)

---

## 3. Feature Distributions by Year

Feature means and variances shift over time, especially for Brine Flowrate, NCG+Steam Flowrate, and Fluid Temperature. This means the model is often asked to extrapolate beyond its training distribution.

**See:** [Feature vs Power Output by Year (Scatter Plots)](feature_vs_power_scatter.html)

**See:** [Power Output Distribution by Year](power_distribution_by_year.html)

---

## 4. Key Findings and Reasoning

- **Temporal Drift:** The relationship between features and power output is not stable. Both the input distributions and their correlations with the target change over time.
- **Model Bias:** Models trained on one year systematically underestimate power output in other years, even as the plant degrades. This is because the model learns a static mapping that does not account for the evolving plant conditions.
- **Why Are Errors Always Positive?**
    - The model, trained on a year with a certain mapping, is presented with future data where the same feature values now correspond to even lower power output (due to degradation or operational changes).
    - However, the feature distributions themselves have shifted, and the model cannot "see" the degradation directly, so it underestimates output for the future, leading to persistent positive errors.
- **Feature-Target Instability:** The instability in feature-target relationships means that even with plant degradation, the model's learned mapping is always a bit "off" in the future.

---

## 5. Recommendations
- **Add explicit temporal features** (e.g., year, time since start, cumulative hours run) to help the model learn the drift.
- **Retrain frequently** or use online learning.
- **Consider regime detection**: If the plant changes operation mode, train separate models for each regime.
- **Add domain knowledge**: If possible, include features that directly measure plant health or degradation.

---

## 6. Plots and Data
- [Feature-Target Correlations Over Time](feature_correlations_over_time.html)
- [Regression Coefficients Over Time](regression_coefficients_over_time.html)
- [Feature vs Power Output by Year (Scatter Plots)](feature_vs_power_scatter.html)
- [Power Output Distribution by Year](power_distribution_by_year.html)
- [Summary Table (CSV)](feature_relationship_summary.csv) 