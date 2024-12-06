# Base Docker image: Anaconda
FROM continuumio/anaconda3

# Install tini
RUN apt-get update && apt-get install -y tini

# Set working directory within the container
WORKDIR /app

# Copy environment.yml
COPY environment.yml /app

# Add necessary channels and update conda
RUN conda update -n base -c defaults conda && conda config --add channels conda-forge

# Install dependencies from environment.yml
RUN conda env create -f environment.yml || conda install --file environment.yml

# Activate environment and install additional pip dependencies
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate mlopsbeadando && pip install flask-restx evidently apache-airflow apache-airflow-providers-smtp"

# Set environment variables
ENV MLFLOW_TRACKING_URI="file:/app/mlruns"
ENV PATH="/opt/conda/envs/mlopsbeadando/bin:$PATH"
ENV AIRFLOW_HOME=/app/airflow

# Copy all other files
COPY . /app

# Copy DAGs
COPY dags /app/airflow/dags

# Set permissions
RUN chmod -R 777 /app/mlruns

# Initialize Airflow database and create admin user
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate mlopsbeadando && airflow db init && airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin"

# Start services using tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate mlopsbeadando && mlflow server --host 0.0.0.0 --port 5102 --backend-store-uri file:/app/mlruns --default-artifact-root /app/mlruns & gunicorn -w 4 -b 0.0.0.0:1080 --timeout 1200 app:app & streamlit run dashboard.py --server.port 8501 & airflow webserver --port 8080 & airflow scheduler"]