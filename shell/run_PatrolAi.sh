#!/bin/bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/root/miniconda3/envs/rapids-22.04/lib/
cd /data/PatrolAi/patrol_ai/python_codes
/root/miniconda3/envs/rapids-22.04/bin/python util_patrol_server.py

