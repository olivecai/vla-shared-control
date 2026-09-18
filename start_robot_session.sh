#!/usr/bin/env bash
# Brings up the full teleop/recording session (container + all ROS nodes + record.py) in a
# single tmux session, one window per terminal from vla/README.md's manual instructions.
#
# Usage:
#   ./start_robot_session.sh
#   IP_ADDRESS=192.168.1.50 CAM_ID=8 ./start_robot_session.sh   # override defaults
#
# Re-running while the session is still up just re-attaches to it instead of restarting
# everything. Detach with `Ctrl-b d`; kill the whole session with `tmux kill-session -t kinova`.
set -euo pipefail

SESSION="kinova"
CONTAINER="vla_shared_control"
IP_ADDRESS="${IP_ADDRESS:-192.168.1.127}"
CAM_ID="${CAM_ID:-0}"
JOY_DEV="${JOY_DEV:-/dev/input/js0}"
JOY_HZ="${JOY_HZ:-30}"
HZ="${HZ:-5}"

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

# window_cmd NAME COMMAND -- opens a docker exec shell in a new window/pane and runs COMMAND in it.
window_cmd() {
    local name="$1" cmd="$2"
    tmux send-keys -t "$SESSION:$name" "docker exec -it $CONTAINER bash" C-m
    sleep 1
    tmux send-keys -t "$SESSION:$name" "$cmd" C-m
}

tmux new-session -d -s "$SESSION" -n bringup
window_cmd bringup "roslaunch kortex_bringup kortex_bringup.launch ip_address:=$IP_ADDRESS"

tmux new-window -t "$SESSION" -n camera
window_cmd camera "rosrun robot_mainframe camera_node.py _cam_id:=$CAM_ID"

tmux new-window -t "$SESSION" -n state
window_cmd state "rosrun robot_mainframe robot_state_node.py"

tmux new-window -t "$SESSION" -n driver
window_cmd driver "rosrun robot_mainframe driver_subscriber.py"

tmux new-window -t "$SESSION" -n joy
# _autorepeat_rate: joy_node defaults to 0 (only publishes /joy on a value CHANGE), so holding
# a stick at a constant deflection stops /joy entirely until the value moves again -- with
# nothing refreshing the downstream cartesian_velocity stream, the arm can stall out between
# updates. Setting a steady republish rate keeps commands flowing continuously while held.
window_cmd joy "rosrun joy joy_node _dev:=$JOY_DEV _autorepeat_rate:=$JOY_HZ"

tmux new-window -t "$SESSION" -n publisher
window_cmd publisher "rosrun robot_mainframe driver_publisher.py"

tmux new-window -t "$SESSION" -n record
window_cmd record "python3 /home/user/vla/record.py --out-dir /home/user/vla/data --cam-ids $CAM_ID --hz $HZ"

tmux select-window -t "$SESSION:bringup"
tmux attach -t "$SESSION"
