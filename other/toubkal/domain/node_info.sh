#!/bin/bash

echo "==== System Information ===="

# Hostname and SLURM node name
echo "Hostname: $(hostname)"
echo "SLURM Node: $SLURMD_NODENAME"

# CPU info
echo "---- CPU Info ----"
echo "CPU Model: $(lscpu | grep 'Model name' | awk -F: '{print $2}' | xargs)"
echo "CPU Cores per Socket: $(lscpu | grep 'Core(s) per socket' | awk '{print $4}')"
echo "Sockets: $(lscpu | grep 'Socket(s)' | awk '{print $2}')"
echo "Total CPU Cores: $(nproc)"
echo "Max Frequency (MHz): $(lscpu | grep 'CPU max MHz' | awk -F: '{print $2}' | xargs)"
echo "Current Frequency (MHz): $(lscpu | grep 'CPU MHz' | awk -F: '{print $2}' | xargs | head -n1)"

# RAM info
echo "---- Memory Info ----"
total_mem=$(free -h | grep Mem: | awk '{print $2}')
echo "Total RAM: $total_mem"

# Optional: GPU info if available
if command -v nvidia-smi &> /dev/null; then
  echo "---- GPU Info ----"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi
