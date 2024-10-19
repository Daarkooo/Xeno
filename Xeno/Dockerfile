# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY app.py .

# Expose the port your app runs on
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "app.py"]
