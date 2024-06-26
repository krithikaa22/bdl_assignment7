# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed pac# FROM python:3.9

# # Set working directory
# WORKDIR /app

# # Copy the requirements file into the container at /app
# COPY requirements.txt .

# # Install dependencies
# RUN pip install -r requirements.txt

# # Copy the current directory contents into the container at /app
# COPY . .

# Use a smaller base image
FROM python:3.9.1-slim

# Set work directory
WORKDIR /usr/src/app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libssl-dev libffi-dev libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /usr/src/app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy project files
COPY . /usr/src/app/
