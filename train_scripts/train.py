from __future__ import print_function

import argparse
import os
import pandas as pd

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

#to make it easy we use an algorithm that has a predict method, as fit_predict seems not to work with the standard sklearn setup of AWS

from sklearn.cluster import AffinityPropagation

# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


if __name__ == '__main__':
    
    # All of the cluster parameters are sent as arguments
    # when this script is executed, during a training job
    parser = argparse.ArgumentParser()

    # Here we set up an argument parser to easily access the parameters
    parser.add_argument('--max_iter', type=int, default=400)
    parser.add_argument('--convergence_iter', type=int, default=15)
    parser.add_argument('--random_state', type=int, default=2)
    parser.add_argument('--damping', type=float, default=0.5)
    parser.add_argument('--preference', type=float, default=-1.0)
    #0 is behavior < 0.23 scikit previously random_state 0 was hard coded
  
    # SageMaker parameters, like the directories for output data and saving models; set automatically
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file and fit data
    data_dir = args.data_dir
    data = pd.read_csv(os.path.join(data_dir, "starbucks_imputed_scaled.csv"), header=None, names=None, skiprows=1)
     # delete unwanted shit from AWS
    col = "Unnamed: 0"
    if col in data.columns:
        data.drop(columns=[col], inplace=True)
        
    model = AffinityPropagation()
    model.fit(data)
    #sagemaker_session.upload_data(path=f"{data_dir}/{pd.to_pickle(model, 'model_file.pkl')}")
       
    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

