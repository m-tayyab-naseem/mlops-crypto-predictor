FROM astrocrpublic.azurecr.io/runtime:3.1-5

# Install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER astro

# Copy project files
COPY scripts /usr/local/airflow/scripts
COPY .env /usr/local/airflow/.env
COPY .dvc /usr/local/airflow/.dvc

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/airflow"