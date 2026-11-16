@echo off
setlocal EnableExtensions

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
cd /d "%ROOT_DIR%"

echo ============================================================
echo   VitalAI: Intelligent Disease Prediction ^& Triage System
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    pause
    exit /b 1
)

if not exist "%ROOT_DIR%\backend\artifacts\nb_model.npz" (
    echo [INFO] Trained model not found. Training Multinomial Naive Bayes model...
    python -m backend.ml.train_model --data-dir data --model-type multinomial --alpha 0.1
    if errorlevel 1 (
        echo [ERROR] Model training failed. Ensure data\Training.csv exists.
        pause
        exit /b 1
    )
)

set "MODE=%~1"
if "%MODE%"=="" (
    echo Choose how to launch VitalAI:
    echo   [1] Backend API + Flutter Chrome Web ^(Recommended / Fastest^)
    echo   [2] Backend API + Flutter Windows Desktop
    echo   [3] Backend API + Flutter Default Device ^(Android/Emulator^)
    echo   [4] Backend API Only ^(http://127.0.0.1:8000/docs^)
    echo.
    set /p "CHOICE=Enter choice [1-4] (default 1): "
    if "%CHOICE%"=="" set "CHOICE=1"
    if "%CHOICE%"=="1" set "MODE=chrome"
    if "%CHOICE%"=="2" set "MODE=windows"
    if "%CHOICE%"=="3" set "MODE=default"
    if "%CHOICE%"=="4" set "MODE=api"
)

if "%MODE%"=="" set "MODE=chrome"

python -m backend.api.healthcheck --host 127.0.0.1 --port 8000
if errorlevel 1 (
    echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
    start "VitalAI FastAPI Backend" /D "%ROOT_DIR%" cmd /k python -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
    echo       Waiting for backend /health to respond...
    python -m backend.api.healthcheck --host 127.0.0.1 --port 8000 --wait 15
    if errorlevel 1 (
        echo [ERROR] Backend failed to respond on http://127.0.0.1:8000/health.
        echo         Check the "VitalAI FastAPI Backend" window for details.
        pause
        exit /b 1
    )
    echo       Backend ready!
) else (
    echo [INFO] FastAPI backend is already running on http://127.0.0.1:8000
)

if /I "%MODE%"=="api" (
    echo [INFO] Backend running at http://127.0.0.1:8000/docs
    start "" "http://127.0.0.1:8000/docs"
    goto :eof
)

where flutter >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Flutter SDK was not found in PATH.
    echo         Backend is still running at http://127.0.0.1:8000/docs
    pause
    exit /b 1
)

echo [2/2] Starting Flutter Frontend (%MODE%)...
cd /d "%ROOT_DIR%\frontend"

if /I "%MODE%"=="chrome" (
    call flutter run -d chrome
) else if /I "%MODE%"=="windows" (
    call flutter run -d windows
) else (
    call flutter run
)

if errorlevel 1 (
    echo.
    echo [WARN] Flutter exited with an error. Check output above.
    pause
)

endlocal
