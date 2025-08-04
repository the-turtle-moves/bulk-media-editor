@echo off
setlocal

REM Record start time and create a default file if needed
set start_time=%TIME%

IF EXIST "C:\Users\Lenovo\Documents\GitHub\image_maker\dist\gui.exe" (
    DEL /F /Q "C:\Users\Lenovo\Documents\GitHub\image_maker\dist\gui.exe"
    ECHO gui.exe has been deleted successfully.
)

echo Starting build process...

for /f "delims=" %%i in ('python -c "import mediapipe; import os; print(os.path.dirname(mediapipe.__file__))" ') do (
    set "mediapipe_path=%%i"
)

pyinstaller --onefile --windowed --add-data "config.json;." --add-data "TikTokSans-VariableFont_opsz,slnt,wdth,wght.ttf;." --add-data "%mediapipe_path%;mediapipe" --upx-dir . gui.py

REM --- Calculate and Display Runtime ---
set end_time=%TIME%

REM Convert start and end times to total centiseconds
set /a h1=%start_time:~0,2%, m1=%start_time:~3,2%, s1=%start_time:~6,2%, cs1=%start_time:~9,2%
set /a h2=%end_time:~0,2%, m2=%end_time:~3,2%, s2=%end_time:~6,2%, cs2=%end_time:~9,2%
set /a start_cs=(h1*360000)+(m1*6000)+(s1*100)+cs1
set /a end_cs=(h2*360000)+(m2*6000)+(s2*100)+cs2

REM Calculate duration, handling midnight crossover
set /a duration_cs=end_cs-start_cs
if %duration_cs% lss 0 set /a duration_cs+=8640000

REM Convert to minutes and seconds for display
set /a runtime_m=duration_cs/6000
set /a runtime_s=(duration_cs%%6000)/100
if %runtime_s% lss 10 set runtime_s=0%runtime_s%

echo.
echo ===========================================
echo Total Runtime: %runtime_m% minutes, %runtime_s% seconds
echo ===========================================
echo.
powershell -c "[console]::beep()"