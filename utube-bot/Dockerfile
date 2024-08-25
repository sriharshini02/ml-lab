# Use the base Python image
FROM python:3.12

# Install system dependencies including sqlite3
RUN apt-get update && \
    apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    ffmpeg

# Install pysqlite3-binary to ensure the correct version of SQLite
RUN pip install pysqlite3-binary

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV GEMINI_API_KEY='AIzaSyDG1Key2SaOs73YXzBQyZ0kxUKH-Liosis'

# Run the application
CMD ["streamlit", "run", "bot.py"]
