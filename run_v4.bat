@echo off
setlocal enabledelayedexpansion
title DJI RoboMaster EP - Assessment 2 (v4) Launcher ^& Real-Time Debugger
color 0B

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:MENU
cls
echo ==============================================================================
echo     DJI RoboMaster EP - Assessment 2 (v4): 360 Sweep ^& Flat Cubic Pursuit
echo          No Strafe (vy=0)  ^|  Dual ToF Auto-ID  ^|  Parallel Alignment
echo ==============================================================================
echo.
echo   [1] Run Physical Robot (Wi-Fi AP 192.168.2.1, Tag 3) + Open Real-Time GUI
echo   [2] Run Physical Robot (USB RNDIS 192.168.42.2, Tag 3) + Open Real-Time GUI
echo   [3] Run Physical Robot (Choose Tag T1/T2/T3 ^& Mode)  + Open Real-Time GUI
echo   [4] Run WSL Simulation Mode (360 Sweep + Dual ToF)    + Open Real-Time GUI
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
start "" python debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (Wi-Fi AP Mode, Target: Tag 3)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' ap 3 false 0.38"
goto DONE

:RUN_RNDIS
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (USB RNDIS Mode, Target: Tag 3)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' rndis 3 false 0.38"
goto DONE

:RUN_CUSTOM
set /p "C_CONN=Connection mode (ap / rndis) [ap]: "
if "%C_CONN%"=="" set "C_CONN=ap"
set /p "C_TAG=Target AprilTag ID (1, 2, or 3) [3]: "
if "%C_TAG%"=="" set "C_TAG=3"
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (Mode: %C_CONN%, Target: Tag %C_TAG%)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' %C_CONN% %C_TAG% false 0.38"
goto DONE

:RUN_SIM
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (Simulation Mode, Target: Tag 3)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' ap 3 true 0.38"
goto DONE

:RUN_GUI
echo [*] Launching Real-Time Debug GUI...
python debug_gui.py
goto MENU

:DONE
echo.
echo ==============================================================================
echo Mission finished or stopped. Press any key to return to menu...
pause >nul
goto MENU
