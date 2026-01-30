@echo off
setlocal
cls

echo ====================================================
echo   ZKYHAXPY BUILD AND PUBLISH TOOL
echo ====================================================

:: 1. Cleanup old files
echo [1/4] Cleaning old build artifacts...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist zkyhaxpy.egg-info rmdir /s /q zkyhaxpy.egg-info

:: 2. Build the package
echo [2/4] Building Wheel and Source Dist...
python -m build
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b %errorlevel%
)

:: 3. Check with Twine
echo [3/4] Checking distribution with Twine...
twine check dist/*
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Twine check failed!
    pause
    exit /b %errorlevel%
)

:: 4. Upload
echo [4/4] Uploading to PyPI...
echo Note: If you haven't set up .pypirc, enter __token__ as username.
twine upload dist/*

echo.
echo ====================================================
echo   PROCESS COMPLETE!
echo ====================================================
pause