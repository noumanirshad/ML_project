import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import os
from src.ML_Project.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Notebooks/Datasets', 'Preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_objects(self):
        '''This function is responsible for data transformation.'''
        logging.info("Get data transformation starting...")
        try:
            num_columns =  ["writing score", "reading score"]
            cat_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
            # num_columns = X.select_dtypes(exclude="object").columns
            # cat_columns = X.select_dtypes(include="object").columns


            num_pipelines = Pipeline(steps=[
                ("imputer" , SimpleImputer(strategy = "median")),
                ('scalar', StandardScaler())
            ])
            
            cat_pipelines = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy = "most_frequent")),
                ('one_hot_encoder', OneHotEncoder()),
                ('scalar', StandardScaler(with_mean=False))
            ])


            logging.info(f"Catogorical Columns: {cat_columns}")
            logging.info(f"Numerical Columns: {num_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipelines", num_pipelines, num_columns),
                ("cat_columns", cat_pipelines, cat_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self,  train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            train_df["average_score"] = (train_df["math score"] + train_df["reading score"]+train_df["writing score"])/3
            test_df["average_score"] = (test_df["math score"] + test_df["reading score"]+ test_df["writing score"])/3

            logging.info("Reading train and test files...")

            preprocessor_obj = self.get_data_transform_objects()

            target_column_name="average_score"
            numerical_columns= ["writing score", "reading score", "math score"]
            ## divide the train dataset to independent and dependent feature
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1) 
            target_feature_train_df=train_df [target_column_name]
            ## divide the test dataset to independent and dependent feature
            input_feature_test_df=test_df.drop (columns=[target_column_name], axis = 1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying processing training and test dataframe")

            input_feature_train_arr= preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            train_arr = np.c_[
            input_feature_train_arr 
            # np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr
                            #  , np.array(target_feature_test_df)
                            ]
            logging.info(f"Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info(f"Data_transformation Successfully Loaded--------------")

            return (
                train_arr,
                test_arr,
                target_feature_train_df,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        



        except Exception as e:
            raise CustomException(e, sys)

