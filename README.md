# BLAEQ_Public
The code to our submitted paper to SIGMOD 2025 titled BLAEQ: Efficient on-GPU Spatial Range Query of Large-scale CFD Mesh using BLAS

The code is fully implemented in Python, including three small datasets for instant running.

# Requirements
The following requirements must be met:
Hardware:
    An NVIDIA GPU supporting CUDA.

Environment:

    -Python 3.x
    
    -cupy
    
    -numpy
    
    -cupyx
    
    -scipy
    
    -pickle (for loading default data set)
    
    -tqdm (for better display)

# HOW TO USE
Simply drag all the .py files and the Data folder into a Python project and compile 'BLAEQ_Public.py'. The algorithm should start running with the default configuration.
For details, plz refer to the comments within the code or our submitted paper.

# APPENDIX
Due to the restriction of file size. We were unable to release all the 9 datasets used in the paper onto github. Therefore, we only put the smallest 3 datasets into the \Data folder. The other files are released on Google Drive:
https://drive.google.com/drive/folders/1rCV5jibO3h8tq00xPw3rj2sBz3GRKltZ?usp=drive_link

To use the other datasets, just put them into the \Data folder and change the 'case' variable in BLAEQ_Public.py with the appropriate file name.
