@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

ECHO ############################################################
ECHO #           ScaleX Automatic Installation Script           #
ECHO ############################################################
ECHO.
ECHO This script will guide you through setting up the ScaleX environment.
ECHO Please ensure Anaconda or Miniconda is installed.
ECHO.

REM --- Check if requirements.txt exists ---
IF NOT EXIST "requirements.txt" (
    ECHO ERROR: requirements.txt not found in the current directory.
    ECHO Please run this script from the root of the ScaleX project folder.
    PAUSE
    EXIT /B 1
)

REM --- Conda Environment Name and Python Version ---
SET ENV_NAME=ScaleX
SET PYTHON_VERSION=3.12

REM --- Check if Conda is accessible ---
CALL conda info >nul 2>nul
IF ERRORLEVEL 1 (
    ECHO WARNING: 'conda' command not found directly.
    ECHO This might happen if Conda is not in your system PATH or initialized for this shell.
    ECHO Attempting to find common Conda activate script...
    ECHO.
    SET "CONDA_BASE_PATH_USER=%USERPROFILE%\Anaconda3"
    SET "CONDA_BASE_PATH_PROGRAMDATA=%ProgramData%\Anaconda3"
    SET "CONDA_BASE_PATH_MINICONDA_USER=%USERPROFILE%\Miniconda3"
    SET "CONDA_BASE_PATH_MINICONDA_PROGRAMDATA=%ProgramData%\Miniconda3"

    SET "ACTIVATE_SCRIPT="
    IF EXIST "%CONDA_BASE_PATH_USER%\Scripts\activate.bat" (
        SET "ACTIVATE_SCRIPT=%CONDA_BASE_PATH_USER%\Scripts\activate.bat"
    ) ELSE IF EXIST "%CONDA_BASE_PATH_PROGRAMDATA%\Scripts\activate.bat" (
        SET "ACTIVATE_SCRIPT=%CONDA_BASE_PATH_PROGRAMDATA%\Scripts\activate.bat"
    ) ELSE IF EXIST "%CONDA_BASE_PATH_MINICONDA_USER%\Scripts\activate.bat" (
        SET "ACTIVATE_SCRIPT=%CONDA_BASE_PATH_MINICONDA_USER%\Scripts\activate.bat"
    ) ELSE IF EXIST "%CONDA_BASE_PATH_MINICONDA_PROGRAMDATA%\Scripts\activate.bat" (
        SET "ACTIVATE_SCRIPT=%CONDA_BASE_PATH_MINICONDA_PROGRAMDATA%\Scripts\activate.bat"
    )

    IF DEFINED ACTIVATE_SCRIPT (
        ECHO Found Conda activate script at: !ACTIVATE_SCRIPT!
    ) ELSE (
        ECHO ERROR: Could not automatically find Conda's activate.bat.
        ECHO Please ensure Conda is installed correctly and its Scripts directory is accessible,
        ECHO or add Conda to your PATH and re-run.
        ECHO You might need to run 'conda init cmd.exe' in an Anaconda Prompt first.
        PAUSE
        EXIT /B 1
    )
    ECHO.
)


REM --- Create Conda Environment ---
ECHO Creating Conda environment: %ENV_NAME% with Python %PYTHON_VERSION%...
CALL conda create -n %ENV_NAME% python=%PYTHON_VERSION% -y
IF ERRORLEVEL 1 (
    ECHO ERROR: Failed to create the Conda environment.
    ECHO It might already exist, or there was an issue with your Conda installation.
    ECHO If it already exists, you might want to remove it first with: conda env remove -n %ENV_NAME%
    PAUSE
    EXIT /B 1
)
ECHO Environment %ENV_NAME% created successfully.
ECHO.

REM --- Activate Environment and Install PyTorch ---
ECHO Activating environment: %ENV_NAME%
IF DEFINED ACTIVATE_SCRIPT (
    CALL "!ACTIVATE_SCRIPT!" %ENV_NAME%
) ELSE (
    CALL conda activate %ENV_NAME%
)
IF ERRORLEVEL 1 (
    ECHO ERROR: Failed to activate the Conda environment.
    PAUSE
    EXIT /B 1
)
ECHO Environment %ENV_NAME% activated.
ECHO.

ECHO --- PyTorch Installation ---
ECHO Choose PyTorch installation type:
ECHO 1. CPU only (Recommended if you don't have a compatible NVIDIA GPU)
ECHO 2. GPU (NVIDIA CUDA - You will need to provide the correct command)
ECHO.
CHOICE /C 12 /M "Enter your choice (1 or 2):"
SET PYTORCH_CHOICE=%ERRORLEVEL%

IF "%PYTORCH_CHOICE%"=="1" (
    ECHO Installing PyTorch for CPU...
    CALL conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    IF ERRORLEVEL 1 (
        ECHO ERROR: Failed to install PyTorch (CPU). Please check your internet connection or Conda setup.
        PAUSE
        EXIT /B 1
    )
) ELSE IF "%PYTORCH_CHOICE%"=="2" (
    ECHO To install PyTorch for GPU (NVIDIA CUDA):
    ECHO 1. Visit the PyTorch official website: https://pytorch.org/get-started/locally/
    ECHO 2. Select your OS (Windows), Package (Conda), Language (Python), and your CUDA version.
    ECHO 3. Copy the generated 'conda install ...' command.
    ECHO.
    ECHO Example for CUDA 11.8: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ECHO Example for CUDA 12.1: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    ECHO.
    SET /P "PYTORCH_GPU_COMMAND=Paste the PyTorch GPU install command here and press Enter: "
    IF DEFINED PYTORCH_GPU_COMMAND (
        ECHO Installing PyTorch for GPU with your command:
        ECHO !PYTORCH_GPU_COMMAND!
        CALL !PYTORCH_GPU_COMMAND!
        IF ERRORLEVEL 1 (
            ECHO ERROR: Failed to install PyTorch (GPU) using your command.
            ECHO Please ensure the command was correct and your CUDA setup is complete.
            PAUSE
            EXIT /B 1
        )
    ) ELSE (
        ECHO No command entered. Skipping GPU PyTorch installation. You will need to install it manually.
    )
) ELSE (
    ECHO Invalid choice. Skipping PyTorch installation. You will need to install it manually.
)
ECHO.

REM --- Install other requirements ---
ECHO Installing other dependencies from requirements.txt...
CALL pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO ERROR: Failed to install packages from requirements.txt.
    ECHO Please check the error messages above.
    PAUSE
    EXIT /B 1
)
ECHO Dependencies installed successfully.
ECHO.

ECHO ############################################################
ECHO #              Installation Complete!                      #
ECHO ############################################################
ECHO.
ECHO To use ScaleX:
ECHO 1. Open a new Anaconda Prompt or Command Prompt.
ECHO 2. Activate the environment: conda activate %ENV_NAME%
ECHO 3. Navigate to the ScaleX directory.
ECHO 4. Run: python inference_scalex.py --help
ECHO.
PAUSE
EXIT /B 0