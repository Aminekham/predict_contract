FROM python:3.10

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt install -y build-essential python3-dev python3-pip
# Set working directory
WORKDIR /app
# Clone TensorFlow source code
RUN git clone https://github.com/tensorflow/tensorflow.git /tensorflow

# Change to TensorFlow source code directory
WORKDIR /tensorflow

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app

# Copy the rest of your application files
COPY . .

# Run your application
CMD ["python", "API2.0.py"]
