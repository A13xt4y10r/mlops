from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import mlflow
from mlflow.tracking import MlflowClient
import sys

# Hozzáadjuk az /app könyvtárat a Python útvonalhoz
sys.path.append('/app')

from MLModel import MLModel  # Importáljuk az MLModel osztályt

# Konfigurációk
TRAIN_ENDPOINT = "http://127.0.0.1:1080/model/train"
FILE_PATH = "/app/data/payment_dataset.csv"
EMAIL_FROM = "airflowtest1155@gmail.com"
EMAIL_TO = "a13xt4y10r@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "airflowtest1155@gmail.com"
SMTP_PASSWORD = "xxxx xxxx xxxx xxxx"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'train_and_compare_model',
    default_args=default_args,
    description='Train and compare ML models daily',
    schedule_interval='0 3 * * *',  # Minden nap 3:00-kor fut
    start_date=days_ago(1),
    tags=['ml'],
)

def train_model():
    with open(FILE_PATH, 'rb') as f:
        files = {'file': f}
        response = requests.post(TRAIN_ENDPOINT, files=files, timeout=120)  # 120 másodperces időkorlát
    if response.status_code == 200:
        result = response.json()
        print("Training result:", result)  # Ellenőrizzük a választ
        # Feltételezzük, hogy a result tartalmazza a szükséges értékeket
        if 'test_accuracy (MAPE)' in result:
            return {'mape_test': result['test_accuracy (MAPE)']}
        else:
            raise KeyError("Response does not contain 'test_accuracy (MAPE)'")
    else:
        raise Exception("Model training failed")

def compare_models(**context):
    client = MlflowClient()
    experiment_name = "my_experiment"
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Ha a "my_experiment" kísérlet nem létezik, hozd létre
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Legutóbbi "train_" kezdetű futás lekérdezése
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName LIKE 'train_%'",
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        # Ha nincs előző futás, automatikusan frissítjük a modellt
        context['task_instance'].xcom_push(key='model_update', value=True)
        return

    last_run = runs[0]
    last_mape_test = last_run.data.metrics["mape_test"]

    # Új modell MAPE értékének lekérdezése
    new_mape_test = context['task_instance'].xcom_pull(task_ids='train_model')['mape_test']

    # Modellek összehasonlítása
    if new_mape_test < last_mape_test:
        # Az új modell jobb, frissítjük a modellt
        context['task_instance'].xcom_push(key='model_update', value=True)
        
        # Új modell mentése a megadott helyre
        model_uri = f"runs:/{context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')}/model"
        mlflow.register_model(model_uri, "best_model")
        
        # Új modell mentése az MLModel.py fájlban meghatározott helyre
        new_model = mlflow.sklearn.load_model(model_uri)
        MLModel.save_model(new_model, 'artifacts/models/randomforest_model.pkl')
        print("New model registered as best_model and saved locally")
    else:
        context['task_instance'].xcom_push(key='model_update', value=False)
        print("Current model is not better than the latest model")

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_USER, SMTP_PASSWORD)
    text = msg.as_string()
    server.sendmail(EMAIL_FROM, EMAIL_TO, text)
    server.quit()

def notify_training_success(**context):
    result = context['task_instance'].xcom_pull(task_ids='train_model')
    subject = "A napi Model-tanítás sikeres"
    body = f"A napi Model-tanítás sikeres. Eredmény: {result}"
    send_email(subject, body)

def notify_model_update(**context):
    model_update = context['task_instance'].xcom_pull(task_ids='compare_models', key='model_update')
    if model_update:
        subject = "Model Update Feljegyzés"
        body = "A model frissítése a teszteredmények alapján megtörtént."
        send_email(subject, body)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

compare_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_models,
    provide_context=True,
    dag=dag,
)

notify_training_task = PythonOperator(
    task_id='notify_training_success',
    python_callable=notify_training_success,
    provide_context=True,
    dag=dag,
)

notify_update_task = PythonOperator(
    task_id='notify_model_update',
    python_callable=notify_model_update,
    provide_context=True,
    dag=dag,
)

train_task >> compare_task >> notify_training_task
compare_task >> notify_update_task