# Use official Python 3.13 slim image
FROM python:3.13.7-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the Flask port
EXPOSE 5002

# Run the Flask app
CMD ["python", "gemini25pro.py"]
