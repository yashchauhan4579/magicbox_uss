#!/bin/bash
# Mass deploy head-pipeline-v3 to all MagicBox devices
TARBALL="/tmp/head_deploy_final.tar.gz"
DEVICE_LIST="${DEVICE_LIST:-/tmp/magicbox_devices.txt}"
LOG_DIR="/tmp/deploy_logs"
PARALLEL=10
SERVER_IP="10.100.0.37"
SERVER_PORT="9010"

mkdir -p $LOG_DIR
> $LOG_DIR/success.txt
> $LOG_DIR/failed.txt
> $LOG_DIR/skipped.txt

SKIP="10.100.1.177 10.100.0.177 10.100.0.125 10.100.0.133"

# Pre-create config file to SCP
cat > /tmp/_head_pipeline_config.json << JSONEOF
{
  "report_url": "http://${SERVER_IP}:${SERVER_PORT}/api/magicbox-crowd/ingest",
  "model_path": "/userdata/linaro/head_deploy/models/best_head_fp16_320.rknn",
  "input_size": 320,
  "cycle_interval_sec": 5,
  "rediscovery_interval_sec": 60,
  "conf_threshold": 0.2,
  "iou_threshold": 0.4,
  "thermal_throttle_c": 80,
  "usscore_url": "http://localhost:8080"
}
JSONEOF

TOTAL=$(wc -l < $DEVICE_LIST)
echo "=== MASS DEPLOY: $TOTAL devices ==="
echo "Started: $(date)"
COUNT=0
RUNNING=0

while read -r IP <&3; do
    # Skip already deployed
    SKIP_THIS=0
    for S in $SKIP; do
        if [ "$IP" = "$S" ]; then SKIP_THIS=1; break; fi
    done
    if [ $SKIP_THIS -eq 1 ]; then
        echo "$IP" >> $LOG_DIR/skipped.txt
        COUNT=$((COUNT + 1))
        echo "[$COUNT/$TOTAL] $IP — SKIPPED"
        continue
    fi

    COUNT=$((COUNT + 1))
    MYCOUNT=$COUNT

    # Deploy in background
    (
        SS="sshpass -p wiredleap12**"
        SO="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

        # Step 1: SCP tarball + config
        if ! $SS scp $SO "$TARBALL" linaro@$IP:/home/linaro/head_deploy.tar.gz </dev/null >/dev/null 2>&1; then
            echo "$IP SCP_FAIL" >> $LOG_DIR/failed.txt
            echo "[$MYCOUNT/$TOTAL] $IP — FAILED (SCP)"
            exit 1
        fi
        $SS scp $SO /tmp/_head_pipeline_config.json linaro@$IP:/home/linaro/head_pipeline.json </dev/null >/dev/null 2>&1

        # Step 2: Extract, place config, install service, start — all in one SSH
        $SS ssh $SO linaro@$IP "echo 'wiredleap12**' | sudo -S bash -c '
            mkdir -p /userdata/linaro/head_deploy
            cd /userdata/linaro/head_deploy
            tar xzf /home/linaro/head_deploy.tar.gz
            rm -f /home/linaro/head_deploy.tar.gz
            mkdir -p /usr/local/uss
            mv /home/linaro/head_pipeline.json /usr/local/uss/head-pipeline.json
            cp /userdata/linaro/head_deploy/head-pipeline-v3.service /etc/systemd/system/
            systemctl daemon-reload
            systemctl enable head-pipeline-v3 2>/dev/null
            systemctl restart head-pipeline-v3
        '" </dev/null >/dev/null 2>&1

        # Step 3: Verify (model load takes 10-15s)
        sleep 15
        STATUS=$($SS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no linaro@$IP \
            "systemctl is-active head-pipeline-v3" </dev/null 2>/dev/null)

        if [ "$STATUS" = "active" ]; then
            echo "$IP" >> $LOG_DIR/success.txt
            echo "[$MYCOUNT/$TOTAL] $IP — SUCCESS"
        else
            echo "$IP SVC_FAIL" >> $LOG_DIR/failed.txt
            echo "[$MYCOUNT/$TOTAL] $IP — FAILED (service: $STATUS)"
        fi
    ) &

    RUNNING=$((RUNNING + 1))

    # Throttle
    if [ $RUNNING -ge $PARALLEL ]; then
        wait -n 2>/dev/null || wait
        RUNNING=$((RUNNING - 1))
    fi

done 3< $DEVICE_LIST

wait

echo ""
echo "=== DEPLOY COMPLETE — $(date) ==="
S=$(wc -l < $LOG_DIR/success.txt)
F=$(wc -l < $LOG_DIR/failed.txt)
K=$(wc -l < $LOG_DIR/skipped.txt)
echo "Success: $S"
echo "Failed:  $F"
echo "Skipped: $K"
if [ -s $LOG_DIR/failed.txt ]; then
    echo ""
    echo "=== FAILED DEVICES ==="
    cat $LOG_DIR/failed.txt
fi
