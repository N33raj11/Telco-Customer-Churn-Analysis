# Telecom Customer Churn — Model Card

## Objective
Predict customer churn using the processed telecom customer dataset.

## Models evaluated
- Logistic Regression
- Random Forest
- Gradient Boosting
- Extra Trees
- KNN

## Model selection
The Session 4 winner was selected using the highest mean 5-fold stratified cross-validation ROC-AUC.

Selected model: **Logistic Regression**

Session 4 mean CV AUC: **0.8472**

## Hyperparameter tuning
RandomizedSearchCV was applied to the selected model using 5-fold CV and ROC-AUC scoring.

Tuned CV AUC: **0.8481**

## Final held-out test evaluation
The test set was evaluated once after tuning.

| Metric | Score |
|---|---:|
| Accuracy | 0.8006 |
| Precision | 0.6598 |
| Recall | 0.5134 |
| F1 | 0.5774 |
| ROC-AUC | 0.8453 |

AUC >= 0.80: **PASS**

## Explainability
SHAP explanations were generated and saved to:
- `outputs/shap_summary.png`
- `outputs/shap_feature_importance.png`

## Saved model
`models/final/best_churn_model.pkl`

The saved pipeline was reloaded and its predictions were verified against the original final-model predictions.

## Limitations
The model is an analytical prediction tool, not a guarantee of individual customer behavior. Performance can change on future or materially different customer populations.
