# Causal Inference & Utilities

## Core Inference Functions

* **`countergrove.inference.predict_ate_binary(model, features)`** /
  **`predict_ate_continuous(model, features, treatment_grid, ...)`**: Predicts
  the Average Treatment Effect (ATE) for binary or continuous treatment, given a
  trained model and input features. Unit-level effects are available via the
  corresponding `predict_ute_binary`/`predict_ute_continuous`.
* **`countergrove.utils.split_y_pred(preds, n_outcome_pred_cols, n_treatment_pred_cols)`**:
  Splits combined model predictions into (outcome params, predictive state
  weights, treatment params). `countergrove.utils.get_column_layout(model)` reads
  `n_outcome_pred_cols`/`n_treatment_pred_cols` off a compiled model instead of
  hardcoding them.
* **`countergrove.bootstrap`**: Offers non-parametric bootstrap estimation methods for
  measuring uncertainty and generating confidence intervals for ATE estimates.

## Example Notebooks

* `notebooks/countergrove_minimal_working_example.ipynb`: Basic workflow for ATE
  estimation.
* `notebooks/countergrove_demo.ipynb`: In-depth end-to-end demonstrations covering
  multiple edge cases.
