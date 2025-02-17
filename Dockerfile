# Use the official Python image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the application code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI application
CMD ["python", "src/mlflow_project/api.py"]
