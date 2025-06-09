FROM public.ecr.aws/neuron/neuron-rtd:2.15.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml/code

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy model serving code
COPY model.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the entrypoint
ENTRYPOINT ["python3", "model.py"] 