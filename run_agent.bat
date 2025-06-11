@echo off
pushd "%~dp0"
python train.py %*
popd
