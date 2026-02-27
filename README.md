BRAINAGEPREDICTION_WITHGRAPHS
Prior to bias correction
The script performs the following steps:
1.  Loads structural connectomes, clinical metadata, gene expression PCA components, and regional brain features (FA, MD, Volume)
2.  Filters and keeps only healthy subjects (excludes AD and MCI)
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
