#!/usr/bin/env bash
# ==============================================================================
# run_wsl.sh - Build & Launch rm_flatness_nav (v4) in WSL (ROS 2 Humble)
# ==============================================================================

CONN_TYPE="${1:-ap}"
TARGET_TAG="${2:-3}"
USE_SIM="${3:-false}"
V_MAX="${4:-0.38}"
PROJECT_DIR="/mnt/d/phd/sce672 - advanced robotics/project apriltag v4"

echo "======================================================================"
echo "[*] Ensuring WSL2 Network Interfaces to RoboMaster EP are UP..."
echo "======================================================================"
sudo -n ip link set eth0 up 2>/dev/null || true
sudo -n ip link set eth1 up 2>/dev/null || true
sudo -n ip link set eth2 up 2>/dev/null || true

TARGET_IP="192.168.2.1"
if [ "$CONN_TYPE" = "rndis" ]; then
    TARGET_IP="192.168.42.2"
fi

if [ "$USE_SIM" != "true" ]; then
    echo "[*] Checking network route to RoboMaster EP at ${TARGET_IP}..."
    if ping -c 1 -W 1 "${TARGET_IP}" >/dev/null 2>&1; then
        echo "[+] Robot is reachable at ${TARGET_IP} (0% packet loss)."
    else
        echo "[!] Route to ${TARGET_IP} needs repair. Scanning mirrored interfaces..."
        SUBNET_PREFIX="${TARGET_IP%.*}."
        IFACE=$(ip -4 -o addr show | grep -F "${SUBNET_PREFIX}" | awk '{print $2}' | head -n1)
        if [ -n "$IFACE" ]; then
            sudo -n ip link set "$IFACE" up 2>/dev/null
            sudo -n ip route replace "${SUBNET_PREFIX}0/24" dev "$IFACE" 2>/dev/null
            echo "[+] Configured direct route on interface ${IFACE}."
        fi
        if ping -c 1 -W 1 "${TARGET_IP}" >/dev/null 2>&1; then
            echo "[+] Successfully established connection to ${TARGET_IP}!"
        else
            echo "[!] WARNING: Robot at ${TARGET_IP} is not responding to ping."
            echo "    Ensure Windows Wi-Fi is connected to 'RMEP-...' before launching."
        fi
    fi
fi

echo "======================================================================"
echo "[*] Syncing & Building ROS 2 Package rm_flatness_nav (v4) in WSL..."
echo "======================================================================"
source /opt/ros/humble/setup.bash
mkdir -p ~/robomaster_ws/src
rm -rf ~/robomaster_ws/src/rm_flatness_nav
cp -r "${PROJECT_DIR}/rm_flatness_nav" ~/robomaster_ws/src/rm_flatness_nav

cd ~/robomaster_ws
colcon build --packages-select rm_flatness_nav --symlink-install

source ~/robomaster_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/raian/.local/lib/python3.10/site-packages:${PYTHONPATH}"

echo "======================================================================"
echo "[+] Launching rm_flatness_nav v4 (conn=${CONN_TYPE}, tag=T${TARGET_TAG}, sim=${USE_SIM}, vmax=${V_MAX} m/s)"
echo "======================================================================"
ros2 launch rm_flatness_nav flatness_pursuit.launch.py \
    conn_type:="${CONN_TYPE}" \
    target_tag_id:="${TARGET_TAG}" \
    use_sim:="${USE_SIM}" \
    v_max_mps:="${V_MAX}" \
    output_dir:="${PROJECT_DIR}"
