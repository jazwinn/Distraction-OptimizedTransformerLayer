@echo off
REM Runs a command with MSVC (cl.exe) on PATH, e.g.:
REM   cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py
REM
REM torch.utils.cpp_extension needs cl.exe for the host-side glue even though
REM nvcc handles the device code, and cl.exe is not on PATH in a plain shell.
REM No setlocal/endlocal on purpose: this is the sole command in a one-shot
REM cmd.exe /c call, so the environment must stay live for %*.

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 >NUL
if errorlevel 1 (
    echo [devenv] vcvarsall.bat failed
    exit /b 1
)

cd /d "%~dp0.."
%*
