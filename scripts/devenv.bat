@echo off
REM Runs a command with MSVC (cl.exe) on PATH, e.g.:
REM   cmd.exe /c scripts\devenv.bat python torch_transformer_benchmark.py
REM
REM torch.utils.cpp_extension needs cl.exe for the host-side glue even though
REM nvcc handles the device code, and cl.exe is not on PATH in a plain shell.
REM No setlocal/endlocal on purpose: this is the sole command in a one-shot
REM cmd.exe /c call, so the environment must stay live for %*.
REM
REM Two things are deliberately not hard-coded:
REM
REM   * the Visual Studio location, which vswhere reports. Installs land under
REM     different year/edition folders, and a wrong guess here reads as "the
REM     CUDA extension cannot be built" rather than as a bad path.
REM
REM   * the MSVC toolset. VS ships several side by side and defaults to the
REM     newest, which CUDA 13.0 rejects outright ("unsupported Microsoft Visual
REM     Studio version") or, worse, accepts far enough that cudafe++ dies with
REM     an access violation. VCVARS_VER pins a toolset nvcc knows; override it
REM     if your CUDA version wants a different one.

if "%VCVARS_VER%"=="" set VCVARS_VER=14.44

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [devenv] vswhere.exe not found -- is Visual Studio installed?
    exit /b 1
)

set "VSPATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -products * -latest -property installationPath`) do set "VSPATH=%%i"
if "%VSPATH%"=="" (
    echo [devenv] vswhere found no Visual Studio installation
    exit /b 1
)

set "VCVARSALL=%VSPATH%\VC\Auxiliary\Build\vcvarsall.bat"
if not exist "%VCVARSALL%" (
    echo [devenv] no vcvarsall.bat under "%VSPATH%" -- install the
    echo          "Desktop development with C++" workload
    exit /b 1
)

call "%VCVARSALL%" x64 -vcvars_ver=%VCVARS_VER% >NUL
if errorlevel 1 (
    echo [devenv] vcvarsall.bat failed for toolset %VCVARS_VER%
    echo          installed toolsets:
    dir /b "%VSPATH%\VC\Tools\MSVC"
    echo          re-run with e.g.  set VCVARS_VER=14.43
    exit /b 1
)

cd /d "%~dp0.."
%*
