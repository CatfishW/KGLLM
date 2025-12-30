#!/bin/bash
# Check if sshpass is installed, if so use it
# sshpass -p Clb1997521 ssh -R 72026:localhost:8000 lobin@vpn.agaii.org -N
# Fallback to manual password entry if needed or just standard ssh
/home/benwulab/anaconda3/envs/KGLLM/bin/python tunnel.py
