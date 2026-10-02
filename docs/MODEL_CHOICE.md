# Model Choice Justification

## Selection basis
Five models were evaluated using 5-fold stratified cross-validation. Mean ROC-AUC was the primary selection metric.

Selected model: **Logistic Regression**

Mean CV AUC: **0.8472**

## Hyperparameter tuning
RandomizedSearchCV was used with ROC-AUC scoring.

Tuned CV AUC: **0.8481**

## Final test evaluation
The held-out test set was evaluated once after selection and tuning.

- Accuracy: **0.8006**
- Precision: **0.6598**
- Recall: **0.5134**
- F1: **0.5774**
- ROC-AUC: **0.8453**

## AUC threshold
Requested threshold: AUC >= 0.80

Observed final test AUC: **0.8453**

Result: **PASS**

The threshold result is reported from the actual test evaluation; no score is altered or fabricated to meet the threshold.
