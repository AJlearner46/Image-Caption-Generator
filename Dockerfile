# Use an official Python runtime as the base image
FROM python:3.8-slim


# Set the working directory in the container
WORKDIR /app

# Copy necessary files into the Docker image
COPY tokenizer.pkl /app/tokenizer.pkl
COPY best_model.h5 /app/best_model.h5

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from tensorflow.keras.applications import VGG16; VGG16(weights='imagenet')"

# Copy the current directory contents into the container at /app
COPY . /app


# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "web_app.py"]