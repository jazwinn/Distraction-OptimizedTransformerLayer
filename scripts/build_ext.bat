@echo off
REM Builds csrc/fused_attention.cu and reports whether it loaded.
REM Run from anywhere:  cmd.exe /c scripts\build_ext.bat

call "%~dp0devenv.bat" python -c "import kernel_ext; m = kernel_ext.get_kernels(verbose=True); print('[build_ext] OK ->', m.__file__) if m else (print('[build_ext] FAILED:', kernel_ext.load_error()), exit(1))"
