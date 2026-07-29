@echo off
REM Build script for rubidium-train on Windows
REM Requires: CMake, CUDA Toolkit, cuDNN

echo ============================================================
echo RUBIDIUM TRAIN - CUDA/C++ Training Engine
echo ============================================================

REM Check for CUDA
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit.
    exit /b 1
)

REM Check for CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: cmake not found. Please install CMake.
    exit /b 1
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure
echo Configuring with CMake...
cmake .. -DCMAKE_CUDA_COMPILER=nvcc -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (
    echo CMake configure failed!
    exit /b 1
)

REM Build
echo Building...
cmake --build . --config Release
if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

echo ============================================================
echo Build successful!
echo Run: build\Release\rubidium-train.exe [corpus_path]
echo ============================================================
