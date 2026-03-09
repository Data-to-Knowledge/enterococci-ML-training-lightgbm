# Development Backlog

This document captures planned improvements and experiments for the Enterococci predictive model. Items are grouped by priority and include context on why each matters.

As of the 2025/26 bathing season (November 2025 to present), the model is performing at approximately 80% sensitivity, specificity, and overall accuracy across roughly 500 samples at 13 monitored sites. This is the baseline we are iterating from.

## High Priority

### 1. Retrain with 2024/25 and 2025/26 season data

The model's been running on data up to late 2024, so the longer we leave it the more it drifts from what's actually happening out there. Environmental patterns shift year to year (different rainfall patterns, land use changes, new infrastructure), so keeping the training data current is the single highest-impact thing we can do. This is also a great first task for Josh as it touches the full pipeline from data ingestion through to evaluation.

### 2. Hyperparameter tuning

We're still on baseline params (the v2 attempt got reverted because it overpredicted exceedances). Worth doing properly with Optuna and our TSCV folds rather than manual tweaking. The TSCV setup already prevents data leakage, so the infrastructure is there.

### 3. Sample weighting for exceedance events

Exceedances are rare so the model tends to under-predict them, which is why we have the precautionary threshold at 140. LightGBM supports sample weighting out of the box through the `sample_weight` parameter in its Dataset, so it's a quick experiment with potentially big upside for sensitivity. Try this before more complex approaches like synthetic oversampling (SMOTE). The goal is to improve sensitivity without destroying specificity.

### 4. Re-introduce Cass Bay and Governers Bay

These were dropped because we couldn't get 50% sensitivity. With fresh data, tuned params, and maybe the meta-learner switched on, worth another crack at getting them over the line for next season. The pipeline already handles multi-site modelling via `SITE_NAME` as a categorical feature, so the main work is data preparation, training, and per-site evaluation.

### 5. Cass Bay data cleaning and second site

We want to filter out dry weather exceedances from Cass Bay's training data, as these are likely driven by a different contamination mechanism that muddies the model. Need to define what constitutes a "dry weather exceedance" (e.g. no significant rainfall in the preceding 24-48 hours). Also want to include the second Cass Bay site (Mid Beach alongside Jetty) to give the model more to work with for that area.

## Medium Priority

### 6. Meta-learner experimentation

It's already built into the pipeline, just switched off in config (`meta_learner: false` in `main_config.yaml`). Could help with site-specific corrections, especially for Cass Bay and Governers Bay. The risk is overfitting on small per-site sample sizes, but the 3-fold TSCV should catch that.

### 7. SHAP and feature importance analysis

Should do this before adding any new features so we know what's actually driving predictions and where the gaps are. Running SHAP on the current model will show which features contribute most overall and, critically, which features the model relies on for exceedance predictions specifically.

### 8. New features (weather, UV, infrastructure)

Worth exploring but guided by SHAP. Some options:

- **Weather beyond rainfall** -- temperature, humidity, and atmospheric pressure may correlate with bacterial survival and growth. These are likely available from the same weather stations already in use and are low-hanging fruit.
- **UV/solar radiation** -- Enterococci are killed by UV exposure, so solar radiation data could help predict die-off rates. However, this may correlate heavily with season (which is already a feature), so the marginal gain needs testing.
- **Infrastructure data** -- number of on-site wastewater management systems (OWMS), proximity to stormwater pipes, sewage outfalls, housing density. These are mostly static or very slow-changing, so they would act more like site-level encoding than dynamic predictors. Could help differentiate sites but unlikely to improve temporal prediction within a site.
- **Feature combinations** -- interaction features (e.g. rainfall intensity multiplied by wind direction category) may capture nonlinear relationships that LightGBM handles implicitly but could benefit from explicit encoding.

### 9. Hilltop features ablation

The season avg and exceedance rate features (`Site_Season_Average` and `Site_Historical_Exceedance_Rate`) are reliable at inference so no rush to remove them, but worth quantifying their contribution with a quick ablation study (train with and without, compare metrics). If they're not materially improving predictions, removing them simplifies the feature pipeline and reduces a dependency.

### 10. Weather station cost-benefit analysis (Rapaki, NIWA Diamond Harbour, NIWA Duvauchelle)

Worth experimenting with alternative weather station data for sites closest to these stations. Options include using NIWA Diamond Harbour, NIWA Duvauchelle, and ECan/NIWA Rapaki directly, or synthesising Rapaki's data using its own records combined with historic data from LPC sites. This adds operational risk in terms of station failover and extra dependencies, but could improve accuracy for harbour sites that are currently distant from the primary stations. Need to run a proper cost-benefit: compare model performance with and without these stations for the relevant sites, and weigh the accuracy gain against the added complexity and failure points.

## Suggested Onboarding Sequence

For a new team member getting up to speed, these tasks build on each other in a logical progression:

1. **Retrain with new data** -- learns the full pipeline end to end
2. **Hyperparameter tuning** -- learns the model configuration and evaluation framework
3. **Hilltop features ablation** -- learns how to design and interpret experiments
4. **Sample weighting** -- learns how the training process works at a deeper level
5. **SHAP analysis** -- learns model interpretability and guides future feature work

Each step produces measurable results and builds understanding of a different part of the system.

## Notes

- The minimum sensitivity threshold for a site to be included in production is **50%**. This is the bar that Cass Bay and Governers Bay need to clear.
- The precautionary threshold (140 MPN/100mL) exists because the model historically under-predicts exceedances. If sensitivity improves through weighting or tuning, this threshold may need revisiting.
- Any changes to feature engineering must be mirrored in the inference pipeline to avoid training/serving skew.