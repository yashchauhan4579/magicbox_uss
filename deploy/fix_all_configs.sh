#!/bin/bash
# Fix config on all 4 deployed devices to be consistent
# SERVER_IP is the only thing you need to change if the server moves
SERVER_IP="10.100.0.37"
SERVER_PORT="9010"

for DEVICE_IP in 10.100.1.177 10.100.0.177 10.100.0.125 10.100.0.133; do
  echo "=== $DEVICE_IP ==="

  # Detect where the model is on this device
  MODEL_PATH=$(sshpass -p 'wiredleap12**' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no linaro@$DEVICE_IP '
    if [ -f /userdata/linaro/head_deploy/models/best_head_fp16_320.rknn ]; then
      echo /userdata/linaro/head_deploy/models/best_head_fp16_320.rknn
    elif [ -f /userdata/linaro/head_count/models/best_head_fp16_320.rknn ]; then
      echo /userdata/linaro/head_count/models/best_head_fp16_320.rknn
    elif [ -f /usr/local/uss/models/best_head_fp16_320.rknn ]; then
      echo /usr/local/uss/models/best_head_fp16_320.rknn
    else
      echo NOTFOUND
    fi
  ' 2>/dev/null)

  echo "  Model: $MODEL_PATH"

  if [ "$MODEL_PATH" = "NOTFOUND" ]; then
    echo "  SKIP: model not found"
    continue
  fi

  sshpass -p 'wiredleap12**' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no linaro@$DEVICE_IP "echo wiredleap12** | sudo -S bash -c 'cat > /usr/local/uss/head-pipeline.json << CONF
{
  \"report_url\": \"http://${SERVER_IP}:${SERVER_PORT}/api/magicbox-crowd/ingest\",
  \"model_path\": \"${MODEL_PATH}\",
  \"input_size\": 320,
  \"cycle_interval_sec\": 5,
  \"rediscovery_interval_sec\": 60,
  \"conf_threshold\": 0.3,
  \"iou_threshold\": 0.45,
  \"thermal_throttle_c\": 80,
  \"usscore_url\": \"http://localhost:8080\"
}
CONF
echo \"  Config updated\"
systemctl restart head-pipeline-v3
echo \"  Service restarted\"
'" 2>/dev/null
  echo ""
done
echo "=== ALL DONE ==="
