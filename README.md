## Overview and Results:
The following science/engineering solution was a Nanodegree solution and solved over AWS sagemaker and some endpoints while applying reinforcement learning with
[vowpal wabbit](https://vowpalwabbit.org/)

The solution chosen, was a contextual bandit algorithm to forecast probabilities on redeeming coupons over time. The Algorithm was trained on a cluster of users, which have a lot of fulfilled customer journeys. Test cluster has users which have few fulfilled customer journeys. Clusters were equally big and there were no mean differences in user characterstics, yet more tests should have been done. Idea was to train the algo on reco behavior of users who redeemed more coupons, to see which it would recommend users with low coupon usage. Since preferences for coupons change over time, the algo shows, how it changes it's user recommendation on coupons over time, when some coupons had no impact at all.

Clusters were built in different ways just for fun:

manually (like above)
with AWS Sagemaker's kmeans (for showing off some cloud skills)
with an external train file and sklearn affinity clustering (which sucks) also with AWS Sagemaker to show the integration of some external estimator
Missing data was forecasted with ExtraTreesRegressor and ExtraTreesClassifier, and a RandomizedSearchCV already forecasted data was not used for further imputation, but this helped forecasting the missing data and putting users and their journeys in some clusters at least.

Permutation tests where made to be sure that the most important user characteristics would be significantly (and practically) different from each other in terms of mean and median. In this way it is assured that, if clusters where not equal (which they luckily are), every found solution by the algorithm unfortunately should have attributed to the the user characterstics and not the different coupons send by Starbucks.

Benchmarks (no treatment group, no alternative costs or random group comparison) not possible because of setup.


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

