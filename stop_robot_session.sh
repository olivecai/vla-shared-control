#!/usr/bin/env bash
# Stops the tmux session from start_robot_session.sh AND the nodes it started inside the
# container. Killing the tmux session alone leaves them running (killing a `docker exec` client
# does not kill the process it started), which is how stale duplicate nodes pile up.
#
# Usage: ./stop_robot_session.sh
set -uo pipefail

SESSION="kinova"
CONTAINER="vla_shared_control"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# The [x] bracket trick keeps pkill from matching its own command line; -9 because leaked
# roslaunch processes are often stopped and ignore SIGTERM.
for pat in '[r]oslaunch' '[r]osmaster' '[r]osout' '[j]oy_node' '[c]ontrol_robot.py' \
           '[p]osition_log.py' '[r]ecord_vla.py' '[n]odelet' '[k]ortex_arm_driver'; do
    docker exec "$CONTAINER" pkill -9 -f "$pat" 2>/dev/null || true
done

echo "Session stopped. Anything still running in the container:"
docker exec "$CONTAINER" pgrep -af '[c]ontrol_robot|[j]oy_node|[r]ecord_vla|[r]oslaunch|[r]osmaster' || echo "  (nothing)"
