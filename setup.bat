@echo off
echo Setting up USCIS Chatbot...

:: Check for Python 3.10
python --version 2>nul | findstr /C:"Python 3.10" >nul
if errorlevel 1 (
    echo Python 3.10 not found. Please install from https://www.python.org/downloads/release/python-3109/
    echo Make sure to check "Add Python to PATH" during installation
    exit /b 1
)

:: Create virtual environment
python -m venv venv

:: Install dependencies using the venv's pip directly
.\venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools
if errorlevel 1 (
    echo Failed to install base dependencies
    exit /b 1
)

.\venv\Scripts\pip.exe install -r requirements.txt
if errorlevel 1 (
    echo Failed to install project requirements
    exit /b 1
)

:: Check if Ollama is installed
where ollama >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo:
    echo:
    echo WARNING: Ollama not found. Please install from https://ollama.ai/download/windows
    echo After installing, run: ollama pull llama3.1:8b ; ollama pull nomic-embed-text
)

echo.
echo Setup complete! To start using the application:
echo 1. Activate the virtual environment:
echo    .\venv\Scripts\activate
echo 2. Run the application:
echo    python main.py
