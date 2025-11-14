# Multi-stage build for RL-Control-Suite

# Base stage with Python and dependencies
FROM python:3.9-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e ".[dev,docs]"

CMD ["/bin/bash"]

# Production stage
FROM base as production

# Copy only necessary files
COPY setup.py pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install package
RUN pip install --no-cache-dir .

# Create non-root user
RUN useradd -m -u 1000 rluser && chown -R rluser:rluser /app
USER rluser

# Set entrypoint
ENTRYPOINT ["python"]
CMD ["-c", "import rl_control; print('RL-Control-Suite ready!')"]

# Testing stage
FROM development as testing

# Run tests
CMD ["pytest", "tests/", "-v", "--cov=src/rl_control"]
