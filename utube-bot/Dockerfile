# Use an official Python runtime as a parent image
FROM python:3.12

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    ffmpeg

# Install a more recent version of sqlite3 if needed
RUN wget https://www.sqlite.org/2024/sqlite-autoconf-3390200.tar.gz && \
    tar xzf sqlite-autoconf-3390200.tar.gz && \
    cd sqlite-autoconf-3390200 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm -rf sqlite-autoconf-3390200 && \
    rm sqlite-autoconf-3390200.tar.gz

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
