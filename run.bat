@echo off

:: Activate the conda environment for ScaleX
CALL "C:\ProgramData\anaconda3\Scripts\activate.bat" ScaleX

:: Navigate to the ScaleX directory (Change path according to yours)
cd /D C:\AI\ScaleX - Web

:: Run ScaleX
python webui_scalex.py