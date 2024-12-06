from datetime import timedelta, datetime
import json
import numpy as np
import pandas as pd
from flask import jsonify
from constants import BASE_COLUMN_NAME, TARGET_COLUMN_ORIGINAL_NAME, TARGET_COLUMN_NAME, LABELING_COLUMNS, SCALED_MIN, SCALED_MAX
from scipy import stats
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import utils
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging

# Logolási szint beállítása
logging.basicConfig(level=logging.DEBUG)

mlflow.set_tracking_uri("file:///app/mlruns") 
# mlflow.set_tracking_uri("file:///Users/a13x/python/mlopsbeadando/mlruns")
mlflow.set_experiment("my_experiment")

class MLModel:
    def __init__(self):
        self.reload_artifacts()

    def reload_artifacts(self):
        # Load ML artifacts during initialization
        self.min_max_scaler_dict = (MLModel.load_model('artifacts/encoders/min_max_scaler_dict.pkl') 
            if os.path.exists('artifacts/encoders/min_max_scaler_dict.pkl') 
            else print('min_max_scaler_dict.pkl does not exist'))
        self.min_max_scaler_target = (MLModel.load_model('artifacts/encoders/min_max_scaler_target.pkl') 
            if os.path.exists('artifacts/encoders/min_max_scaler_target.pkl') 
            else print('min_max_scaler_target.pkl does not exist'))
        self.label_encoders = (MLModel.load_model('artifacts/encoders/label_encoders_dict.pkl') 
            if os.path.exists('artifacts/encoders/label_encoders_dict.pkl') 
            else print('label_encoders_dict.pkl does not exist'))
        self.model = (MLModel.load_model('artifacts/models/randomforest_model.pkl') 
            if os.path.exists('artifacts/models/randomforest_model.pkl') 
            else print('randomforest_model.pkl does not exist'))

    def predict(self, inference_row, need_preproc):
        """
        Predicts the outcome based on the input data row.

        This method applies the preprocessing pipeline to the input data, performs necessary
        transformations, and uses the preloaded model to make a prediction. The 'V24' column
        is removed from the data frame as part of the preprocessing steps. If an error occurs
        during the prediction process, it catches the exception and returns a JSON object with
        the error message and a 500 status code.

        Parameters:
        - inference_row: A single row of input data meant for prediction. Expected to be a list or
        a series that matches the format and order expected by the preprocessing pipeline and model.

        Returns:
        - On success: Returns the prediction as an integer.
        - On failure: Returns a JSON response object with an error message and a 500 status code.

        Notes:
        - Ensure that the input data row is in the correct format and contains the expected features
        excluding 'V24', which is not required and will be removed during preprocessing.
        - The method is wrapped in a try-except block to handle unexpected errors during prediction.
        """
        try:

            # ha több worker van, az egyiken betanítom, a többi worker nem frissíti az artifact-okat a memóriában
            # ha betanítás után rögtön prediktálok, akkor kerülhetek olyan workerhez, ahol ezek nincsennek még betöltve
            # ezért betöltöm (nyílván jobb workaround, ha train után restartoljuk az appot de fejlesztésnél így egyszerűbb nekem)
            if not self.model:
                self.reload_artifacts()
                print('relaod artifacts')

            #pd dataframe létrehozása az input adatból
            #df = pd.DataFrame(inference_row, index=['invoice_id'])
            df = pd.DataFrame([inference_row])
            #print(df[TARGET_COLUMN_ORIGINAL_NAME])

            # eredeti due_in_date kivétele, még mielőtt nagyon feldolgozom aza datokat
            due_date = pd.to_datetime(df[TARGET_COLUMN_ORIGINAL_NAME], format='%Y%m%d')
            due_date_str = str(due_date.dt.strftime('%Y-%m-%d').iloc[0])

            if need_preproc:
                df = self.preprocessing_pipeline_inference(df)   
            
            df.to_csv('artifacts/preprocessed_data/saved_inference_data.csv', index=False) #előfeldolgozott adat mentése

             # Elindítom az MLflow run-t, egyedi névvel
            with mlflow.start_run(run_name=f"inference_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"):

                #prediktálok
                y_pred = self.model.predict(df)

                #invert scaling
                original_data = self.min_max_scaler_target.inverse_transform([y_pred]) 
                #kerekítem mert az eredtei y-k is egész napok voltak
                rounded_data = round(original_data[0][0])
                print(f"A visszaalakított érték (prediktált): {rounded_data}")

                # prediktált clear date kiszámítása
                predicted_clear_date = due_date + timedelta(days=rounded_data)
                # a prediktált clear date stringgé alakítása megadott formátumban
                clear_date_str = str(predicted_clear_date.dt.strftime('%Y-%m-%d').iloc[0])
                print(f"A clear_date érték (prediktált): {clear_date_str}")

                # Mentés artifactokként
                mlflow.log_artifact('artifacts/preprocessed_data/saved_inference_data.csv')  # Előfeldolgozott adat
                mlflow.log_metric('predicted_clear_date', rounded_data)  # A predikált érték
                mlflow.log_param('due_date_str', due_date_str)  # Eredeti due_date string

           
            #return rounded_data
            return clear_date_str, due_date_str, rounded_data

        except Exception as e:
            return jsonify({'message': 'Internal Server Error. ','error': str(e)}), 500


    def preprocessing_pipeline(self, df):
        """Preprocess the data to handle missing values,
        create new features, encode categorical features, 
        and normalize the data using min max scaling.
        Returns the preprocessed dataframe.
        
        Keyword arguments:
        df -- DataFrame with the data

        Returns:
        df -- DataFrame with the preprocessed data
        """

        folder = 'artifacts/encoders'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/preprocessed_data'
        MLModel.create_new_folder(folder)

        folder = 'artifacts/models'
        MLModel.create_new_folder(folder)
        
        df = utils.drop_unwanted_rows(df)

        # oszlopok eldobása amik sztem nem relevánsak a predikciónál
        df=df.drop('document_create_date.1', axis=1)
        df=df.drop('document_create_date', axis=1)
        df=df.drop('baseline_create_date', axis=1)
        df=df.drop('posting_date', axis=1)
        # a komplett oszlopot is eldobom, amiben csak nan van
        df=df.drop('area_business', axis=1)

        # a due_in_date számként van tárolva, ezt átalakítom, mert fontos adat
        df['due_in_date'] = pd.to_datetime(df['due_in_date'], format='%Y%m%d')

        # dátumok konvertálása datetime formátumba, egységesen, mert 2 féle képpen vannak tárolva
        df['clear_date'] = pd.to_datetime(df['clear_date'])
        df['due_in_date'] = pd.to_datetime(df['due_in_date'])

        # feature engineering
        # kifizetési napok kiszámítása a bázis értékhez viszonyítva
        # ezzel alőáll a target oszlop
        df = utils.create_new_features(df,BASE_COLUMN_NAME, TARGET_COLUMN_ORIGINAL_NAME,TARGET_COLUMN_NAME)

        # label encoding
        self.label_encoders = {}
        new_columns = []
        for col in LABELING_COLUMNS:
            # label encoder betöltése
            encoder = LabelEncoder()
            new_data = encoder.fit_transform(df[col])
            new_columns.extend(col+'_encoded')
            df[col+'_encoded'] = new_data
            self.label_encoders[col] = encoder
        df.drop(columns=LABELING_COLUMNS, inplace=True)

        #df = utils.drop_outlayers(df)

        # skálázok MinMaxScaler-er
        self.min_max_scaler_dict = {}
        for col in df.columns:
            min_max_scaler = MinMaxScaler(feature_range=(SCALED_MIN, SCALED_MAX)) # a 0.01 a MAPE miatt van, mert nem maradhat benne nulla
            df[col] = min_max_scaler.fit_transform(df[[col]])
            self.min_max_scaler_dict[col] = min_max_scaler
            #a target scaler-ének mentése, az inverz scaling miatt
            if col == TARGET_COLUMN_NAME:
                self.min_max_scaler_target = min_max_scaler
                MLModel.save_model(min_max_scaler, 
                           'artifacts/encoders/min_max_scaler_target.pkl')
                
        df.to_csv('artifacts/preprocessed_data/saved_train_data.csv', index=False) #előfeldolgozott adat mentése

        #artifact-ek elmentése
        #TO-DO: verziózás
        MLModel.save_model(self.min_max_scaler_dict, 
                           'artifacts/encoders/min_max_scaler_dict.pkl')
        MLModel.save_model(self.label_encoders, 
                           'artifacts/encoders/label_encoders_dict.pkl')
        

        return df

    def preprocessing_pipeline_inference(self, df):
        """Preprocess the inference row to match
        the features we created for training data.
        Returns the preprocessed dataframe for inference.
        
        Keyword arguments:
        sample_data -- Pandas series with the inference data

        Returns:
        input_df -- DataFrame with the preprocessed inference data
        """
        
        #ez a forráskódi sor lehet nem kell
        #df = df.dropna(subset=['invoice_id'])

        # a komplett oszlopot is eldobom, amiben csak nan van
        df=df.drop('area_business', axis=1, errors='ignore')
        #oszlopok eldobása amik sztem nem relevánsak
        df=df.drop('document_create_date.1', axis=1, errors='ignore')
        df=df.drop('document_create_date', axis=1, errors='ignore')
        df=df.drop('baseline_create_date', axis=1, errors='ignore')
        df=df.drop('posting_date', axis=1, errors='ignore')
        #ez nem kell, ebből targetet számolnám
        df=df.drop('due_in_date', axis=1, errors='ignore')
        df=df.drop('clear_date', axis=1, errors='ignore')

        # label encoding
        for col, encoder in self.label_encoders.items():
            # Csak azokat az értékeket használjuk, amelyeket a betanítási adatokban láttunk
            known_labels = set(encoder.classes_)
            df[col] = df[col].apply(lambda x: x if x in known_labels else 'unknown')
            
            # Az "unknown" címke hozzáadása az encoder osztályaihoz
            if 'unknown' not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, 'unknown')
            
            if df[col].isnull().all():
                df.drop(columns=[col], inplace=True)
            else:
                df = df.dropna(subset=[col])
                new_data = encoder.transform(df[col]) #itt már csak TRANSFORM, nem FIT-eljük
                df[col+'_encoded'] = new_data
        df.drop(columns=LABELING_COLUMNS, inplace=True, errors='ignore')

        # skálázok MinMaxScaler-er
        for col, scaler in self.min_max_scaler_dict.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])

        return df
    
    def get_accuracy_full(self, X, y):
        """
        Calculate and print the overall accuracy of the model using a data set.

        Args:
            X: Features for the data set.
            y: Actual labels for the data set.

        Returns:
            The accuracy of the model on the provided data set.
        """
        y_pred = self.model.predict(X)
        mape_train = mean_absolute_percentage_error(y, y_pred)*100

        print("Accuracy MAPE: ", mape_train)

        return mape_train

    def train_and_save_model(self, df):
        try:
            # Előfeldolgozás
            df = self.preprocessing_pipeline(df)

            y = df[TARGET_COLUMN_NAME]
            X = df.drop(columns=TARGET_COLUMN_NAME)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

            regr = RandomForestRegressor(bootstrap=True, max_depth=20, min_samples_leaf=1, min_samples_split=6, n_estimators=120, random_state=42)

            # MLflow experiment indítása, run név hozzáadása
            run_name = f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            with mlflow.start_run(run_name=run_name):
                regr.fit(X_train, y_train)
                self.save_model(regr, 'artifacts/models/randomforest_model.pkl')  # Elmentjük a modellt

                self.model = regr

                y_pred_train = self.model.predict(X_train)
                y_pred_test = self.model.predict(X_test)

                # Invert scaling
                original_test_target = self.min_max_scaler_target.inverse_transform([y_test])
                predicted_test_target = self.min_max_scaler_target.inverse_transform([y_pred_test])

                mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100
                mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

                # MLflow paraméterek és metrikák logolása
                mlflow.log_param("max_depth", 20)
                mlflow.log_param("n_estimators", 120)
                mlflow.log_metric("mape_train", mape_train)
                mlflow.log_metric("mape_test", mape_test)

                # Modell mentése MLflow-ba
                mlflow.sklearn.log_model(regr, "model")

                return mape_train, mape_test

        except Exception as e:
            mlflow.log_param("error", str(e))
            print(f"Hiba történt: {str(e)}")
            raise e
        finally:
            mlflow.end_run()

    @staticmethod
    def create_new_folder(folder):
        """Create a new folder if it doesn't exist.
        
        Keyword arguments:
        folder -- Path to the folder

        Returns:
        None
        """
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model