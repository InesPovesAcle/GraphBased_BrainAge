**BrainAgePrediction_withgraphs**

Prior to bias correction

The script performs the following steps:
1.  Loads structural connectomes, clinical metadata, gene expression PCA components, and regional brain features (FA, MD, Volume)
2. Filters and keeps **only healthy subjects** (excludes AD and MCI)
3.  Applies connectome preprocessing (percentile thresholding and log transform)
4.  Builds graph representations where:
    - Nodes = brain regions
    - Edges = connectome connections
    - Node features = FA, MD, Volume, clustering coefficient
    - Global features = demographics, graph metrics, and PCA components
5. Trains a GATv2 model using:
   - 7-fold stratified cross-validation
   - 10 repetitions per fold
   -  Early stopping
6. Evaluates performance using MAE, RMSE, and R²
7.  Trains a final model on all healthy subjects
The outputs are saved to: $WORK/ines/results/addecode_training_eval_plots_save/
  - graph_data_list_addecode.pt → processed graph dataset
  - cv_checkpoints/model_fold_*_rep_*.pt → CV model checkpoints
  - cv_predictions_addecode.csv → predicted vs real ages
  - cv_train_loss_mean_95CI.png and cv_validation_loss_mean_95CI.png → training curves
  - model_trained_on_all_healthy.pt → final trained model


**BrainAgePrediction_withbias**

This cript adds to the previous one:

1. Brain Age Bias correction (BAG bias):
    - Fits a linear model BAG ~ Age on the training split for each fold/repeat
    - Applies the correction to the test split → cBAG (fold-wise corrected BAG)
  
2. Global out-of-fold bias correction (recommended for repeated CV):
    - Aggregates predictions per subject across repeats
    - Fits BAG ~ Age using subject-level OOF mean predictions
    - Produces cBAG_global (more reliable bias estimate under repeated CV)

3. Extended exports for reproducibility and downstream analysis:
    - Saves bias coefficients (slope/intercept) per fold×repeat
    - Saves a clean subject-level table (mean/std across repeats)
    - Saves stability tables (MAE/RMSE/R² per fold×repeat + summary)
    - ROC-AUC analysis for APOE4 carrier vs non-carrier using subject-level scores

Outputs are saved to: $WORK/ines/results/results_with_bias_correction/
    - cv_predictions_addecode.csv → predictions + BAG, cBAG, and cBAG_global
    - bias_coefficients_per_fold_repeat.csv → slope/intercept per fold×repeat
    - subject_level_predictions_clean.csv → one row per subject (mean/std across repeats)
    - metrics_per_fold_repeat.csv + stability_summary_fold_repeat.csv → model stability summaries
    - plot_subject_level_pred_vs_real.png, plot_subject_level_cBAG_global_hist.png, plot_stability_mae_hist.png → summary plots
    - model_trained_on_all_healthy_corrected.pt → final model trained on all healthy subjects (bias-correction pipeline)





