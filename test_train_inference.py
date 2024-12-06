import pandas as pd
import numpy as np
from pathlib import Path
from constants import TARGET_COLUMN_NAME
import utils
import logging

# Logging beállítása
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from MLModel import MLModel

def test_prediction_accuracy():
    data_path = 'data/payment_dataset.csv'

    obj_mlmodel = MLModel()
    df = pd.read_csv(data_path)

    # DELETE - csak a fejlesztés miatt, hogy gyorsabban lefusson
    # df = df.head(10)

    # ----------------------------
    # Training pipeline
    # ----------------------------
    df_preprocessed_train = obj_mlmodel.preprocessing_pipeline(df)

    # a közös y
    y_expected = df_preprocessed_train[TARGET_COLUMN_NAME]

    logging.info(f'Number of rows in preprocessed training data: {len(df_preprocessed_train)}')

    # mape számítás az egyben feldolgotott sorokkal
    accuracy_train_pipeline_full = obj_mlmodel.get_accuracy_full(
        df_preprocessed_train.drop(columns=TARGET_COLUMN_NAME), 
        y_expected
    )
    # kerekítem
    accuracy_train_pipeline_full = np.round(accuracy_train_pipeline_full, 2)

    # ----------------------------
    # Inference pipeline
    # ----------------------------
    obj_mlmodel = MLModel()
    df = pd.read_csv(data_path)

    # DELETE - csak a fejlesztés miatt, hogy gyorsabban lefusson
    # df = df.head(10)

    # azon sorok eldobása, amiket a train során is eldobok
    df = utils.drop_unwanted_rows(df) 
    # df = utils.drop_outlayers(df)

    df_preprocessed_inference = pd.DataFrame()
    # egyesével feldolgozom a sorokat
    for i in df.iterrows():
        row_dict = i[1].to_dict()

        df = pd.DataFrame([row_dict])
        preprocessed_df_single = obj_mlmodel.preprocessing_pipeline_inference(df)
        # összevonom az egyesével visszakapot sorokat egy közös df-be
        df_preprocessed_inference = pd.concat([df_preprocessed_inference, preprocessed_df_single], ignore_index=True)

    # mape számítása az egyesével feldolgozott sorokkal
    accuracy_inference_pipeline_full = obj_mlmodel.get_accuracy_full(
        df_preprocessed_inference, 
        y_expected
    )
    
    # kerekítem
    accuracy_inference_pipeline_full = np.round(accuracy_inference_pipeline_full, 2)
   
    # eredmények kiírása
    logging.info(f'Training pipeline accuracy: {accuracy_train_pipeline_full}')
    logging.info(f'Inference pipeline accuracy: {accuracy_inference_pipeline_full}')

    # teszt megfelelőség vizsgálata
    assert accuracy_train_pipeline_full == accuracy_inference_pipeline_full, 'Inference prediction accuracy is not as expected'

if __name__ == "__main__":
    test_prediction_accuracy()