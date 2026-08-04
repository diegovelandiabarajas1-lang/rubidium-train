@echo off
REM ============================================================
REM RUBIDIUM CPU - Build Script (Windows)
REM ============================================================
echo ============================================================
echo Building Rubidium CPU Training Engine
echo ============================================================

REM Create build directory
if not exist build_cpu mkdir build_cpu
cd build_cpu

REM Configure
echo --- Configuring ---
cmake ../src -DCMAKE_CXX_FLAGS="/O2 /openmp /std:c++17"

REM Build
echo --- Building ---
cmake --build . --config Release

echo.
echo ============================================================
echo Build complete!
echo ============================================================
echo.
echo Usage:
echo   rubidium_cpu_train.exe train ^<corpus_dir^>
echo   rubidium_cpu_train.exe finetune ^<base_model^> ^<corpus_dir^>

cd ..
