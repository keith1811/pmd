# dataplatform-ml-demandtrends



## Required packages

All packages needed to run the code in src is included in the `requirerments.txt`.
To make sure that you have all required packages run

```bash
!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
```

for the unit-testing:

```bash
!cp ../unit_tests/local_requirements.txt ~/.
%pip install -r ~/local_requirements.txt
```


## Purchase Order Classification MLOps workflow
Generally, each pipeline has its own seperated configuration file which can be found under "/{root_folder}/conf/pipeline_configs/{pipeline_name}.yml", before running any pipelines the configurations
must be adjusted, then the associated notebook must be executed from "/{root_folder}/notebooks/{pipeline_name}". 
The configs dedicated to each one of the three pipelines contains the following settings:

### Modules

All modules are included in the `/{root_folder}/modules/ `-folder.

### Notebooks 

Notebooks are situated in the `/{root_folder}/notbooks/ ` folder.

### pipeline_configs
1. model_train_feature_creator
               
    **raw_input_data:** 

      1. path to raw community tables (8 tables) 
      2. path to unspsc_reference_table. This table is used to filter only legit UNSPSC codes 
        
    **data_prep_params:** 

      - **unspsc_processsing:** ilter legit unspsc codes based on the given string pattern. Also fitlter out the segments, family, class that contains zeros

      - **handle_str_na_vals:** unify the format 'NAN' string values to one specific format across all columns ,and filter out the rows in which 'CODE' and 'Description' are equal to 'N/A'

      - **select_cols_instructions:** select a subset of columns 

      - **drop_duplicates_instructions:** drop duplicate rows

      - **handle_missing_vals:** drop/fill 'null' values
        
2. model_train
   
    **feature_transformation:** 

      - **feature_concat_instructions:** concact all feature columns except label column 'CODE', to one single column 'CONCAT_FEATURE' column

      - **sampling_instructions:** configuration regarding the sampled data for training 

      - **tokenization_instructions:** tokenization configuration including the number of vocabulary

      - **padding_instructions:** padding configurations, including the length of the array after padding. This one is used to prepare the data shape for the input of the deep learning model
   
    **mlflow_params:** paramters to setup mlflow experiments

    **train_val_split_config:** traing/validation split configuration 

    **train_instructions:** configuration regarding CNN model 
      

**Note:** the steps in the data preparation are similar to data preparation steps for creating the features for training, the the main difference is inference data does not contain label column. Therefore, these steps are excluded from the inference pipeline.

3. model_inference:

    **data_prep_params:**
        
      - **handle_str_na_vals:** unify the format 'NAN' string values to one specific format across all columns ,and filter out the rows in which 'CODE' and 'Description' are equal to 'N/A'

      - **select_cols_instructions:** select a subset of columns 

      - **drop_duplicates_instructions:** drop duplicate rows

      - **handle_missing_vals:** drop/fill 'null' values

     **feature_transformation:** 

      - **feature_concat_instructions:** concact all feature columns except label column 'CODE', to one single column 'CONCAT_FEATURE' column

      - **handeling_imbalanced_label_instructions:**
          3 methods to handle imbalanced dataset:
          1. filter: Drop UNSPSC codes, which has a count less that a specified threshold
          2. replace: Mask UNSPSC codes, which has a count less that a specified threshold and assign class 'other or 99999999' to them
          3. no_action: pass the dataframe the way that it is

      - **sampling_instructions:** configuration regarding the sampled data for training 

      - **tokenization_instructions:** tokenization configuration including the number of vocabulary

      - **padding_instructions:** padding configurations, including the length of the array after padding. This one is used to prepare the data shape for the input of the deep learning model

    **mlflow_params:** the configuration of model registry including model name and stage of the model



#### Environment Variables Configs

  **.base_data_params.env:** intermediate storage path which is common across all environments ( if not this one can be moved to .dev.env)


  **.dev.env:** 

  - inference_input_table_path: path to input inference table
  - inference_input_table_type: type of inference input table

  - inference_output_table_path  = path to output inference table

  - batch_scoring: [False, True]
  - batch_size: size of batch for inference 
  - save_output: [False, True]


## Pipelines 
### 1. model_train_feature_creator: 
    Intermediate feature preparation for training

1. before running the script, go to **"/{root_folder}/conf/pipeline_configs/model_train_feature_creator.yml"** to adjust path to input data and other configurations
2. set the path in **/{root_folder}/conf/.base_data_params.env**, where the intermediate data after processing must be stored 
3. go to **"/{root_folder}/notebooks/model_train_feature_creator"** and run all the cells

### 2. model_train: 
    train, validate and log the model, metrics and charts on Azure Databricks Mlflow

1. before running the script, go to **"/{root_folder}/conf/pipeline_configs/model_train.yml"** to adjust run_name and other configuration for training (default run name is "CNN_baseline_exp")
2. go to **"/{root_folder}/notebooks/model_train"** and select the "env" from dbutils drop down and run all the cells


### 3. model_infernece: 
    Inference and prediction

1. before running the script, go to **"/{root_folder}/conf/pipeline_configs/model_infernece.yml"** to configure inference configuration including the model name and the stage of the registered model which is going to be loaded for prediction
2. go to **/{root_folder}/conf/dev/.dev.env** to select the Input/Output inference table path as well as other configurations such as if the predictions are conducted in batch or all at once (depend on the memory of the cluster and size of the infernece data)
2. go to **"/{root_folder}/notebooks/model_infernece"** and select the "env" from dbutils drop down and run all the cells


## MLFlow
All experiments and artifacts and models can be found under experiment tab from left side bar of Azure databricks. 

## Model Registry
Each time that a training script is executed, the model (including keras fitted model, tokenizer and label encoder) will be registered in azure databricks model registry.
To acess the registered model, you have to select machine learning persona from top, left side bar menu and then click the model registry. There, under '{model_name}' that you specified in 'model_inference.yml', you should be able to find registered models with different versions.

Depending on the mlflow configuration for '/{root_folder}/conf/pipeline_configs/model_inference.yml' namely, 'model_name' and 'model_registry_stage' the model can be loaded for the inference. if the stage of two models are the same, the databricks always take the latest version. 


## Unit Testing:
To initiate unit testing for the module, open notebook “notebooks/run_test_notebooks” and execute all it cells. The requirements of unit testing modules, are separately stored under “/{root_folder}/unit_tests/local_requirements.txt”. 
