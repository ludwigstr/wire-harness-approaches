#!/bin/bash
# usage: continue_after.sh <pid> <gpu> <clip>...: wait for PID to exit, then run stream.sh on the clips
cd .
P=$1; shift
while kill -0 $P 2>/dev/null; do sleep 30; done
echo "previous step (pid $P) finished $(date +%H:%M)"
exec scripts/physworld_mpm_chains/stream.sh "$@"
