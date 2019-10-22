# Use latest Python runtime as a parent image
FROM python:3.6.5-slim



# Meta-data
LABEL maintainer="Aly Sivji <alysivji@gmail.com>" \
      description="Data Science Workflow #1: Self-Contained Container\
      Libraries, data, and code in one image"

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required libraries
RUN pip --no-cache-dir install numpy pandas seaborn sklearn jupyter
#RUN pip install --no-use-wheel --allow-all-external --allow-unverified blinker -r numpy pandas seaborn sklearn jupyter
#RUN pip install numpy pandas seaborn sklearn jupyter --proxy http(s)://proxy:8080 --trusted-host pypi.python.org

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Run jupyter when container launches
ENTRYPOINT  ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
