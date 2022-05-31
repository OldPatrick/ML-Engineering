## Folder and Files needed
    
#### folder: train_scripts
- train.py (external traininf file for using sklearns Affinity Clustering in AWS)

#### folder: starbucks_data
##### original starbucks files
- portfolio.json
- profile.json
- transcript.json (maybe too large)
<br>

##### generated through the notebook code or part of the folder starbucks_data
- starbucks_df_per_coupon.csv 
  --------------------------------(retrieved from: 01_Starbucks_Data_Preparation.ipynb) 
- starbucks_df_per_person.csv 
  ---------------------------------(retrieved from: 01_Starbucks_Data_Preparation.ipynb)   
- starbucks_imputed.csv 
  ---------------------------------------(retrieved from: 02_Starbucks_Data_Imputation_and_Statistical Testing_per_person.ipynb)
- starbucks_df_per_id.csv---------------------------------------(retrieved from: 02_Starbucks_Data_Imputation_and_Statistical Testing_per_person.ipynb) 
- starbucks_imputed_scaled.csv
  -------------------------------(retrieved from: 04_AWS_Internal_Clustering_kmeans.ipynb)
- starbucks_df_per_coupon_imputed.csv
  ----------------------(retrieved from: 06_Map_imputed_data_to_coupon_data.ipynb)

#### folder: excursus_full_affinity_clustering
- Local_Affinity_clustering.ipynb
- labels.csv (forecasted labels of long affinity_clustering)
- scaled data for forecasting (starbucks_imputed scaled again)

#### files:
- 01_Starbucks_Data_Preparation.ipynb
- 02_Starbucks_Data_Imputation_and_Statistical Testing_per_person.ipynb
- 03_Manual_User_Clustering_and_Goal_Derivation.ipynb
- 04_AWS_Clustering_Unsupervised_kmeans.ipynb
- 05_AWS_External_Clustering_Sklearn_Affinity.ipynb
- 06_Map_imputed_data_to_coupon_data.ipynb
- 07a_Final_Model_Contextual_Bandits_with_Vowpal_Wabbit_cover_algorithm.ipynb
- 07b_Final_Model_Contextual_Bandits_with_Vowpal_Wabbit_bagging_algorithm.ipynb

<br>

## Libraries needed
- boto3
- datatable 
- io
- json
- mxnet
- mlxtend
- numpy
- os
- pandas
- plotly
- plotly express
- re
- sagemaker
- sklearn
- vowpalwabbit

