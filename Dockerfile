# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory content into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8501 for streamlit 
EXPOSE 8501

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py"]

