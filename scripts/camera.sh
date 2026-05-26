#!/bin/bash

v4l2-ctl -d /dev/video4 \
  --set-fmt-video=width=640,height=480,pixelformat=MJPG \
  --set-parm=30 \
  --stream-mmap \
  --stream-count=100
