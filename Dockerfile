# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9

# Create a non-root user with a home directory
RUN useradd -m -u 1000 user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY requirements.txt $HOME/app/
RUN chown -R user:user $HOME/app

# Switch to the non-root user before installing dependencies
USER user
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY --chown=user:user . $HOME/app

# Set the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
