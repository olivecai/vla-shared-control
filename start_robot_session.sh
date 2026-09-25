#!/usr/bin/env bash
# Brings up the full teleop/recording session (container + all ROS nodes + record_vla.py) in a
# single tmux session, one window per terminal from the manual instructions in
# kortex_bringup/src/kortex_bringup/record_vla.py's docstring.
#
# Usage:
#   ./start_robot_session.sh
#   IP_ADDRESS=192.168.1.50 CAM_ID=1 SERIAL_NO=123456789 ./start_robot_session.sh   # override defaults
#
# Re-running while the session is still up just re-attaches to it instead of restarting
# everything. Detach with `Ctrl-b d`; kill the whole session with `tmux kill-session -t kinova`.
set -euo pipefail

SESSION="kinova"
CONTAINER="vla_shared_control"
IP_ADDRESS="${IP_ADDRESS:-192.168.1.127}"
CAM_ID="${CAM_ID:-0}"
SERIAL_NO="${SERIAL_NO:-017322072808}"
JOY_DEV="${JOY_DEV:-/dev/input/js0}"
JOY_HZ="${JOY_HZ:-30}"
HZ="${HZ:-5}"
OUT_DIR="${OUT_DIR:-/home/user/la/data}"

if ! command -v tmux >/dev/null; then
    echo "tmux is not installed (apt install tmux)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists -- attaching instead of restarting."
    exec tmux attach -t "$SESSION"
fi

(cd "$SCRIPT_DIR/kinova" && docker compose build && docker compose up -d)

for pat in '[r]oslaunch' '[r]osmaster' '[r]osout' '[j]oy_node' '[c]ontrol_robot.py' \
           '[p]osition_log.py' '[r]ecord_vla.py' '[n]odelet' '[k]ortex_arm_driver'; do
    docker exec "$CONTAINER" pkill -9 -f "$pat" || true
done

# window_cmd NAME COMMAND -- opens a docker exec shell in a new window/pane and runs COMMAND in it.
window_cmd() {
    local name="$1" cmd="$2"
    tmux send-keys -t "$SESSION:$name" "docker exec -it $CONTAINER bash" C-m
    sleep 1
    tmux send-keys -t "$SESSION:$name" "$cmd" C-m
}

tmux new-session -d -s "$SESSION" -n bringup
window_cmd bringup "roslaunch kortex_bringup kortex_bringup.launch ip_address:=$IP_ADDRESS"

tmux new-window -t "$SESSION" -n joy
# _autorepeat_rate: joy_node defaults to 0 (only publishes /joy on a value CHANGE), so holding
# a stick at a constant deflection stops /joy entirely until the value moves again -- with
# nothing refreshing the downstream cartesian_velocity stream, the arm can stall out between
# updates. Setting a steady republish rate keeps commands flowing continuously while held.
window_cmd joy "rosrun joy joy_node _dev:=$JOY_DEV _autorepeat_rate:=$JOY_HZ"

tmux new-window -t "$SESSION" -n control
window_cmd control "rosrun kortex_bringup control_robot.py"

tmux new-window -t "$SESSION" -n camera
window_cmd camera "roslaunch realsense2_camera rs_camera.launch camera:=camera$CAM_ID serial_no:=$SERIAL_NO"

tmux new-window -t "$SESSION" -n undo
window_cmd undo "rosrun kortex_bringup position_log.py"

tmux new-window -t "$SESSION" -n record
# record_vla.py isn't installed as a rosrun-able script (not marked executable, unlike
# control_robot.py), so cd to it and run it with python3 directly instead.
window_cmd record "cd /home/user/kinova/catkin_ws/src/kortex_bringup/src/kortex_bringup && python3 record_vla.py --out-dir $OUT_DIR --hz $HZ --image-topics /camera$CAM_ID/color/image_raw"

tmux select-window -t "$SESSION:bringup"
tmux attach -t "$SESSION"


