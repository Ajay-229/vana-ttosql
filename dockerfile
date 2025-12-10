# Dockerfile

FROM python:3.11-slim

# Add ARG statements to receive the authenticated proxy URL from docker-compose
ARG HTTP_PROXY
ARG HTTPS_PROXY

# Set environment variables for the build steps (Crucial for apt and initial proxy handshake)
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY

WORKDIR /app

# --- START: CERTIFICATE INSTALLATION FIX ---
# 1. Copy the merged corporate root certificate
COPY root.crt /usr/local/share/ca-certificates/root.crt

# 2. Install necessary packages and update system certificate store
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# --- END: CERTIFICATE INSTALLATION FIX ---

# --- START: PIP CONFIGURATION FIX (Timeout Added) ---
# This explicitly tells pip where to find the trusted certificate, sets the proxy, and increases timeout.

# 1. Create the configuration directory
RUN mkdir -p /root/.pip

# 2. Write the pip.conf file
# *** IMPORTANT CHANGE: Added 'timeout = 600' (10 minutes) ***
RUN echo "[global]" >> /root/.pip/pip.conf \
    && echo "cert = /usr/local/share/ca-certificates/root.crt" >> /root/.pip/pip.conf \
    && echo "trusted-host = pypi.org files.pythonhosted.org" >> /root/.pip/pip.conf \
    && echo "proxy = ${HTTP_PROXY}" >> /root/.pip/pip.conf \
    && echo "timeout = 600" >> /root/.pip/pip.conf
# --- END: PIP CONFIGURATION FIX ---

# Install dependencies (This step should now succeed without timeouts)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (The .env file is excluded by .dockerignore)
COPY app.py .
COPY train.py .
COPY Chinook.sqlite .
COPY doc.txt .

# Clear the sensitive proxy environment variables for the final runtime environment
ENV HTTP_PROXY=
ENV HTTPS_PROXY=

# Expose the port where the server will run
EXPOSE 8000

# Command to run the training script first, then start the Gunicorn server
CMD ["sh", "-c", "python train.py && gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:server.app --bind 0.0.0.0:8000"]