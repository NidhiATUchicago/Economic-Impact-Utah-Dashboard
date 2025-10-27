# Use an official Python 3.8 base image
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Shiny app code into the container
COPY app.py /app/app.py

# Expose the port the Shiny app will run on
EXPOSE 8000

# Default command: start the Shiny for Python app
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app.py"]