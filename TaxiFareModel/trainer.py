
# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.data import clean_data,  get_data
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        #time_pipe = Pipeline('time_step',TimeFeaturesEncoder(time_column = "pickup_datetime"))
        #distance_pipe = Pipeline('distance_step',DistanceTransformer(time_column = "pickup_datetime"))

        # create time pipeline
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        # create distance pipeline
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        # Add the model of your choice to the pipeline
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])

        self.pipeline = pipe
        return self.pipeline
    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline().fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        # compute y_pred on the test set
        y_pred = self.pipeline.predict(self.X_test)

        return compute_rmse(y_pred, self.y_test)


if __name__ == "__main__":

    # store the data in a DataFrame
    df = get_data()

    # clean
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    trainer = Trainer(X,y)

    #trainer.set_pipeline()

    trainer.run()


    test = trainer.evaluate()

    print(test)