FROM python:3.12.3

LABEL name="cogs107-pymc-cli"
LABEL description="PyMC 5.21.1 CLI environment"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git-core git mercurial subversion \
    build-essential \
    byobu \
    curl \
    htop \
    libfreetype6-dev \
    libzmq3-dev \
    pkg-config \
    rsync \
    software-properties-common \
    unzip \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # Added dependencies for netcdf4 and pytables (hdf5)
    libnetcdf-dev \
    libhdf5-dev \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up group and user permissions
RUN groupadd -g 1000 jovyan
RUN adduser --disabled-password --gecos '' --uid 1000 --gid 1000 jovyan
# Ensure jovyan owns their home directory and .local for pip installs
RUN chown -R jovyan:jovyan /home/jovyan
RUN echo "jovyan ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to jovyan user
USER jovyan
WORKDIR /home/jovyan

# Set Python path to ensure user installs are preferred
ENV PYTHONPATH="/home/jovyan/.local/lib/python3.12/site-packages:${PYTHONPATH}"
ENV PATH="/home/jovyan/.local/bin:${PATH}"

# Create requirements.txt
RUN <<EOF cat > requirements.txt
# Base scientific stack
numpy==2.2.4
scipy==1.15.2
pandas
matplotlib==3.10.1
seaborn==0.13.2
xarray
# I/O and data formats
netcdf4
tables # PyTables for HDF5
# PyMC and ArviZ
pymc==5.21.1
arviz==0.21.0
# Jupyter environment
jupyterlab
notebook
ipywidgets
jupyter-server-proxy
jupyter-resource-usage
# Utilities
tqdm
EOF

# Install Python packages using pip
# Using --no-cache-dir to reduce image size
# Using --user to install into the user's site-packages
RUN pip install --no-cache-dir --user -r requirements.txt \
    && rm requirements.txt

# Set the final working directory for the container
WORKDIR /home/jovyan/project
RUN mkdir -p /home/jovyan/project # Ensure the final WORKDIR exists

# Preinstall VSCode server components (owned by jovyan)
RUN mkdir -p /home/jovyan/.local/share/code-server/extensions \
             /home/jovyan/.vscode-server/extensions \
             /home/jovyan/.vscode-server-insiders/extensions

# Default to bash
# No conda activate needed anymore
SHELL ["/bin/bash", "-c"]
CMD ["sleep", "infinity"]
