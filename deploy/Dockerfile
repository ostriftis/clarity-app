# ============================
# 1. Base image
# ============================
FROM python:3.10-slim

# Avoid Python writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ============================
# 2. System dependencies
# ============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================
# 3. Workdir
# ============================
WORKDIR /app

# ============================
# 4. Copy requirements + setup
# ============================
COPY requirements.txt .
COPY setup.sh .

# ============================
# 5. Install Python deps
# ============================
RUN pip install --upgrade pip

# Install unsloth (your exact sequence)
RUN bash setup.sh

# ============================
# 6. Copy the rest of the code
# ============================
COPY . .

# ============================
# 7. Default command
# ============================
# You can override this at runtime
ENTRYPOINT ["python", "main.py"]
