FROM tensorflow/tensorflow:latest-gpu

# Set up a working directory for your app
WORKDIR /app

# Copy your project files into the Docker image
COPY requirements.txt .
COPY app/ app/

# Install system-level dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev

# Create a Python virtual environment
RUN python -m venv venv

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip and install your project dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Set the default command for the Docker image
CMD ["python", "app/main.py"]
