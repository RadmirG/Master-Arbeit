# ======================================================================================================================
# A Docker Container to execute FEniCS python library for solwing PDE with FEM methods.
# FEniCS execution is possible only under Linux derivatives. The current container 
# gives all the possibilities to run a python file, which is using FEniCS under Windows.
# ======================================================================================================================
# Radmir Gesler, 2024, master thesis at BHT Berlin by Prof. Dr. Frank Haußer
# ======================================================================================================================


# Use official FEniCS image
FROM quay.io/fenicsproject/stable:current

# Working folder in the container
WORKDIR /app

# Copy the Python file into the container
COPY forward_heat_equa.py /app/

# Update pip to the latest version
RUN pip3 install --no-cache-dir --upgrade pip

# Install all needed libraries
RUN pip3 install --no-cache-dir \
    matplotlib \
    numpy \
    scipy \
    plotly

# Default command to run the Python script
#CMD ["python3", "/app/forward_heat_equa.py"]