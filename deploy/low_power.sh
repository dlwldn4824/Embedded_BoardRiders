#!/bin/bash
# Pi 전력 줄이기. 주행 전에 한 번 실행한다 (재부팅하면 원래대로 돌아간다).
#   bash deploy/low_power.sh          # 적용
#   bash deploy/low_power.sh off      # 되돌리기
# - CPU 최고 클럭을 1.8GHz 로 묶는다 (기본 2.4GHz). 순간 전력이 줄고, 320 모델 YOLO 는 0.35초 안에 끝난다.
# - 쓰지 않는 Wi-Fi 절전은 건드리지 않는다 (VNC 가 끊기면 안 되므로).
set -e
if [ "$1" = "off" ]; then
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    sudo sh -c "cat $c/cpuinfo_max_freq > $c/scaling_max_freq"
  done
  echo "CPU 최고 클럭 원래대로"
else
  for c in /sys/devices/system/cpu/cpu*/cpufreq; do
    sudo sh -c "echo 1800000 > $c/scaling_max_freq"
  done
  echo "CPU 최고 클럭 1.8GHz 로 제한"
fi
echo "현재: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq) kHz  $(vcgencmd get_throttled)  $(vcgencmd pmic_read_adc EXT5V_V 2>/dev/null)"
