@echo off
setlocal enabledelayedexpansion
title DJI RoboMaster EP - Native Windows Assessment 2 (Camera-Only, No WSL)
color 0A

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:MENU
cls
echo ==============================================================================
echo   DJI RoboMaster EP - Native Windows Assessment 2 (Parallel Version, No WSL)
echo          100%% Camera-Based Navigation  ^|  No ToF  ^|  Differential Flatness
echo ==============================================================================
echo.
echo   [1] Run Physical Robot on Windows (Wi-Fi AP 192.168.2.1, Tag 3) + Open GUI
echo   [2] Run Physical Robot on Windows (USB RNDIS 192.168.42.2, Tag 3) + Open GUI
echo   [3] Run Physical Robot on Windows (Choose Tag T1/T2/T3 ^& Mode)  + Open GUI
echo   [4] Run Windows Simulation Mode (Camera-Only, 360-Sweep)        + Open GUI
echo   [5] Open Real-Time Debug GUI Only (debug_gui.py)
echo   [6] Exit
echo.
echo ==============================================================================
set /p "CHOICE=Select option [1-6]: "

if "%CHOICE%"=="1" goto RUN_AP
if "%CHOICE%"=="2" goto RUN_RNDIS
if "%CHOICE%"=="3" goto RUN_CUSTOM
if "%CHOICE%"=="4" goto RUN_SIM
if "%CHOICE%"=="5" goto RUN_GUI
if "%CHOICE%"=="6" exit /b 0
goto MENU

:RUN_AP
echo [*] Checking Wi-Fi connection to RoboMaster EP (192.168.2.1)...
ping -n 1 -w 800 192.168.2.1 >nul 2>&1
if errorlevel 1 (
    echo [!] Robot not responding at 192.168.2.1. Auto-connecting to Wi-Fi 'RMEP-21b3c9'...
    netsh wlan connect name="RMEP-21b3c9" ssid="RMEP-21b3c9" >nul 2>&1
    timeout /t 3 /nobreak >nul
)
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Mission Runner (Mode: Wi-Fi AP, Target: Tag 3)...
python src\mission_runner_win.py --conn ap --tag 3 --vmax 0.38
goto DONE

:RUN_RNDIS
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Mission Runner (Mode: USB RNDIS, Target: Tag 3)...
python src\mission_runner_win.py --conn rndis --tag 3 --vmax 0.38
goto DONE

:RUN_CUSTOM
set /p "C_CONN=Connection mode (ap / rndis) [ap]: "
if "%C_CONN%"=="" set "C_CONN=ap"
set /p "C_TAG=Target AprilTag ID (1, 2, or 3) [3]: "
if "%C_TAG%"=="" set "C_TAG=3"
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Mission Runner (Mode: %C_CONN%, Target: Tag %C_TAG%)...
python src\mission_runner_win.py --conn %C_CONN% --tag %C_TAG% --vmax 0.38
goto DONE

:RUN_SIM
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Simulation Mode (Target: Tag 3)...
python src\mission_runner_win.py --sim --tag 3 --vmax 0.38
goto DONE

:RUN_GUI
echo [*] Launching Real-Time Debug GUI...
python src\debug_gui.py
goto MENU

:DONE
echo.
echo ==============================================================================
echo Mission finished or stopped. Press any key to return to menu...
pause >nul
goto MENU
