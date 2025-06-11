@echo off
pushd "%~dp0"
REM Install required Python packages for KOFBot
pip install numpy matplotlib gymnasium pywin32 pydirectinput torch colorama ReadWriteMemory "ray[rllib]"
popd
