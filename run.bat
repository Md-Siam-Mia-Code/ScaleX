@echo off

:: Activate the conda environment for ScaleX
CALL "C:\ProgramData\<your-anaconda-distribution-name>\Scripts\activate.bat" ScaleX

:: Navigate to the ScaleX directory (Change path according to yours)
cd /D <path-to-your-ScaleX>

:: Run ScaleX
python inference_scalex.py -i Input -o Output -f v1.4 -b x4 -s 4