# Use the official Python 3.9-slim image as the base image
FROM python:3.9-slim

# Update package list and install iperf
RUN apt-get update && apt-get install -y iperf && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY app.py .

# Expose the port your app runs on
EXPOSE 5000

# Command to run your Flask app
CMD ["python", "app.py"]
