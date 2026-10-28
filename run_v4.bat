@echo off
setlocal enabledelayedexpansion
title DJI RoboMaster EP - Assessment 2 (v4 Launcher & Real-Time Debugger)
color 0B

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:MENU
cls
echo ==============================================================================
echo     DJI RoboMaster EP - Assessment 2: 360 Sweep ^& Flat Cubic Pursuit
echo          No Strafe (vy=0)  ^|  Differential Flatness  ^|  Parallel Alignment
echo ==============================================================================
echo.
echo   --- NATIVE WINDOWS RUNNER (No WSL, Camera-Only, Zero Virtualization Latency) ---
echo   [1] Run Physical Robot on Windows (Wi-Fi AP 192.168.2.1, Tag 3) + GUI
echo   [2] Run Physical Robot on Windows (USB RNDIS 192.168.42.2, Tag 3) + GUI
echo   [3] Run Windows Simulation Mode (Camera-Only, Tag 3)            + GUI
echo.
echo   --- WSL / ROS 2 HUMBLE RUNNER (Runs via WSL2 Virtualization) ---
echo   [4] Run Physical Robot in WSL (Wi-Fi AP 192.168.2.1, Tag 3)     + GUI
echo   [5] Run Physical Robot in WSL (USB RNDIS 192.168.42.2, Tag 3)   + GUI
echo   [6] Run WSL Simulation Mode (ROS 2 Humble, Tag 3)               + GUI
echo.
echo   --- TOOLS ---
echo   [7] Open Real-Time Debug GUI Only (debug_gui.py)
echo   [8] Exit
echo.
echo ==============================================================================
set /p "CHOICE=Select option [1-8]: "

if "%CHOICE%"=="1" goto WIN_AP
if "%CHOICE%"=="2" goto WIN_RNDIS
if "%CHOICE%"=="3" goto WIN_SIM
if "%CHOICE%"=="4" goto WSL_AP
if "%CHOICE%"=="5" goto WSL_RNDIS
if "%CHOICE%"=="6" goto WSL_SIM
if "%CHOICE%"=="7" goto RUN_GUI
if "%CHOICE%"=="8" exit /b 0
goto MENU

:WIN_AP
echo [*] Checking Wi-Fi connection to RoboMaster EP (192.168.2.1)...
ping -n 1 -w 800 192.168.2.1 >nul 2>&1
if errorlevel 1 (
    echo [!] Robot not responding at 192.168.2.1. Auto-connecting to Wi-Fi 'RMEP-21b3c9'...
    netsh wlan connect name="RMEP-21b3c9" ssid="RMEP-21b3c9" >nul 2>&1
    timeout /t 3 /nobreak >nul
)
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Mission Runner (Mode: Wi-Fi AP, Camera-Only, Target: Tag 3)...
python src\mission_runner_win.py --conn ap --tag 3 --vmax 0.38
goto DONE

:WIN_RNDIS
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Mission Runner (Mode: USB RNDIS, Camera-Only, Target: Tag 3)...
python src\mission_runner_win.py --conn rndis --tag 3 --vmax 0.38
goto DONE

:WIN_SIM
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching Native Windows Simulation Mode (Camera-Only, Target: Tag 3)...
python src\mission_runner_win.py --sim --tag 3 --vmax 0.38
goto DONE

:WSL_AP
echo [*] Checking Wi-Fi connection to RoboMaster EP (192.168.2.1)...
ping -n 1 -w 800 192.168.2.1 >nul 2>&1
if errorlevel 1 (
    echo [!] Robot not responding at 192.168.2.1. Auto-connecting to Wi-Fi 'RMEP-21b3c9'...
    netsh wlan connect name="RMEP-21b3c9" ssid="RMEP-21b3c9" >nul 2>&1
    timeout /t 3 /nobreak >nul
)
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (Wi-Fi AP Mode, Target: Tag 3)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' ap 3 false 0.38"
goto DONE

:WSL_RNDIS
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (USB RNDIS Mode, Target: Tag 3)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' rndis 3 false 0.38"
goto DONE

:WSL_SIM
echo [*] Starting Real-Time Debug GUI on Windows...
start "" python src\debug_gui.py
echo [*] Launching ROS 2 v4 Node in WSL (Simulation Mode, Target: Tag 3)...
wsl.exe bash -c "sed -i 's/\r$//' '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' && bash '/mnt/d/phd/sce672 - advanced robotics/project apriltag v4/run_wsl.sh' ap 3 true 0.38"
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
