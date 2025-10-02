@echo off
setlocal enabledelayedexpansion

REM Set 7-Zip path (check both common installation locations)
set "SEVENZIP="
if exist "C:\Program Files\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files\7-Zip\7z.exe"
if exist "C:\Program Files (x86)\7-Zip\7z.exe" set "SEVENZIP=C:\Program Files (x86)\7-Zip\7z.exe"

if not defined SEVENZIP (
    echo ERROR: 7-Zip not found!
    echo Please install 7-Zip from https://www.7-zip.org/
    echo Or update the SEVENZIP variable in this script with your 7-Zip path.
    pause
    exit /b 1
)

python -m venv venv

set "ERROR_OCCURRED=0"

call venv\Scripts\activate

python.exe -m pip install --upgrade pip

pip install -r requirements.txt

pip install --no-deps git+https://github.com/microsoft/MoGe.git
pip install moge==2.0.0
pip install xfuser==0.4.4
pip install scipy click gradio matplotlib trimesh
pip install git+https://github.com/openai/CLIP.git

pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

REM Extract files in data_engine folder
echo Extracting data_engine\Metric3D.7z...
"%SEVENZIP%" x "data_engine\Metric3D.7z" -o"data_engine\" -y
if !errorlevel! neq 0 (
    echo ERROR: Failed to extract Metric3D.7z
    set "ERROR_OCCURRED=1"
) else (
    echo Successfully extracted Metric3D.7z
)
echo.

echo Extracting data_engine\MoGe.7z...
"%SEVENZIP%" x "data_engine\MoGe.7z" -o"data_engine\" -y
if !errorlevel! neq 0 (
    echo ERROR: Failed to extract MoGe.7z
    set "ERROR_OCCURRED=1"
) else (
    echo Successfully extracted MoGe.7z
)
echo.

echo Extracting data_engine\vggt.7z...
"%SEVENZIP%" x "data_engine\vggt.7z" -o"data_engine\" -y
if !errorlevel! neq 0 (
    echo ERROR: Failed to extract vggt.7z
    set "ERROR_OCCURRED=1"
) else (
    echo Successfully extracted vggt.7z
)
echo.

REM Extract file in main directory
echo Extracting dep_whls.7z...
"%SEVENZIP%" x "dep_whls.7z" -o"." -y
if !errorlevel! neq 0 (
    echo ERROR: Failed to extract dep_whls.7z
    set "ERROR_OCCURRED=1"
) else (
    echo Successfully extracted dep_whls.7z
)
echo.

if !ERROR_OCCURRED! equ 1 (
    echo.
    echo ========================================
    echo ERROR: One or more extractions failed!
    echo .7z files will NOT be deleted.
    echo ========================================
    pause
    exit /b 1
)

REM All extractions successful - delete .7z files
echo.
echo All extractions completed successfully!
echo Deleting .7z files...
echo.

del "data_engine\Metric3D.7z"
del "data_engine\MoGe.7z"
del "data_engine\vggt.7z"
del "dep_whls.7z"

pip install dep_whls/flash_attn-2.7.4+cu128torch2.7-cp310-cp310-win_amd64.whl

pip install git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38

pip install dep_whls/deepspeed-0.17.5+e1560d84-2.7torch+cu128-cp310-cp310-win_amd64.whl

deactivate
echo Installation complete.
pause
