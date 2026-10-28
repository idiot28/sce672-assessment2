@echo off
setlocal
title DJI RoboMaster EP - Autonomous Flatness Pursuit and Teleop
color 0A

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ==============================================================================
echo     DJI RoboMaster EP - Assessment 2: Wi-Fi Teleop and Autonomous Pursuit
echo          No Strafe vy=0  ^|  Differential Flatness  ^|  Target: Tag 3
echo ==============================================================================
echo.

set "PY_CMD=C:\Users\raian\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY_CMD%" set "PY_CMD=py -3.12"

echo [*] Using Python interpreter: %PY_CMD%
echo [*] Checking Wi-Fi route to RoboMaster EP at 192.168.2.1...

ping -n 1 -w 800 192.168.2.1 >nul 2>&1
if errorlevel 1 goto CONNECT_WIFI
echo [+] Robot is online and responding at 192.168.2.1 - 0 percent packet loss.
goto LAUNCH_APP

:CONNECT_WIFI
echo [!] Robot not responding at 192.168.2.1.
echo [*] Auto-connecting Windows Wi-Fi to RMEP-21b3c9...
netsh wlan connect name="RMEP-21b3c9" ssid="RMEP-21b3c9" >nul 2>&1
timeout /t 3 /nobreak >nul
ping -n 1 -w 1200 192.168.2.1 >nul 2>&1
if errorlevel 1 (
    echo [WARN] Robot did not respond to ping at 192.168.2.1.
    echo        Ensure your robot is turned on and PC is connected to the robot AP.
) else (
    echo [+] Successfully connected to RoboMaster EP.
)

:LAUNCH_APP
echo.
echo ==============================================================================
echo [*] Launching Real-Time GUI directly in Wi-Fi AP Mode...
echo     - Mode: Live Video Stream 30 FPS native, zero-latency in-memory blit
echo     - Initial State: TELEOP - Use WASD / Arrow keys to position robot
echo     - Spacebar: Trigger Autonomous Differential Flatness Pursuit to Tag 3
echo ==============================================================================
echo.

"%PY_CMD%" src\debug_gui.py --conn ap --tag 3 --vmax 0.38

if errorlevel 1 (
    echo.
    echo [ERROR] Controller exited with an error code.
    echo Press any key to close...
    pause >nul
)

exit /b 0
