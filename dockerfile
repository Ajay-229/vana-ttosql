# Dockerfile

# Changed to the full python:3.11 image (larger, but more complete)
FROM python:3.11-slim

# Add ARG statements to receive the authenticated proxy URL from docker-compose
ARG HTTP_PROXY
ARG HTTPS_PROXY

# Set the environment variables for the build steps (Crucial for pip)
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY

WORKDIR /app

# Install dependencies (pip will use the ENV variables set above)
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