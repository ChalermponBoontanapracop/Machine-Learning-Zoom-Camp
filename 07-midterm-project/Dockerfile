
# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Copy dependency files
COPY Pipfile Pipfile.lock ./

# Install dependencies inside the container
RUN pipenv sync --system

# Copy Flask application, Historical API and trained model
COPY predict.py historical.py model.bin ./

# Copy Web Interface
COPY web/ ./web/

# Copy Historical Electricity Dataset
COPY data/ ./data/

# Expose API port
EXPOSE 5000

# Run Flask API using Waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "predict:app"]
