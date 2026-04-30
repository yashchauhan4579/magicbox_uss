#!/bin/bash
# Deploy to remaining devices using xargs for reliable parallelism
TARBALL="/tmp/head_deploy_final.tar.gz"
CONFIG="/tmp/_head_pipeline_config.json"
LOG_DIR="/tmp/deploy_logs"

# Add failed devices to remaining list for retry
for FAIL_IP in $(awk '{print $1}' $LOG_DIR/failed.txt 2>/dev/null); do
    grep -q "^${FAIL_IP}$" /tmp/remaining_devices.txt || echo "$FAIL_IP" >> /tmp/remaining_devices.txt
done
# Reset failed log
> $LOG_DIR/failed.txt

TOTAL=$(wc -l < /tmp/remaining_devices.txt)
echo "=== DEPLOYING TO $TOTAL REMAINING DEVICES (10 parallel) ==="
echo "Started: $(date)"

deploy_device() {
    IP=$1
    SS="sshpass -p wiredleap12**"
    SO="-o ConnectTimeout=15 -o StrictHostKeyChecking=no"

    # SCP tarball
    if ! $SS scp $SO "$TARBALL" linaro@$IP:/home/linaro/head_deploy.tar.gz </dev/null >/dev/null 2>&1; then
        echo "$IP SCP_FAIL" >> /tmp/deploy_logs/failed.txt
        echo "$IP — FAILED (SCP)"
        return 1
    fi

    # SCP config
    $SS scp $SO "$CONFIG" linaro@$IP:/home/linaro/head_pipeline.json </dev/null >/dev/null 2>&1

    # Extract + install + start
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

    # Verify (15s for model load)
    sleep 15
    STATUS=$($SS ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no linaro@$IP \
        "systemctl is-active head-pipeline-v3" </dev/null 2>/dev/null)

    if [ "$STATUS" = "active" ]; then
        echo "$IP" >> /tmp/deploy_logs/success.txt
        echo "$IP — SUCCESS"
    else
        echo "$IP SVC_FAIL" >> /tmp/deploy_logs/failed.txt
        echo "$IP — FAILED (service: $STATUS)"
    fi
}

export -f deploy_device
export TARBALL CONFIG

cat /tmp/remaining_devices.txt | xargs -P 10 -I {} bash -c 'deploy_device "$@"' _ {}

echo ""
echo "=== BATCH COMPLETE — $(date) ==="
echo "Total success: $(wc -l < /tmp/deploy_logs/success.txt)"
echo "Total failed:  $(wc -l < /tmp/deploy_logs/failed.txt)"
echo "Total skipped: $(wc -l < /tmp/deploy_logs/skipped.txt)"
if [ -s /tmp/deploy_logs/failed.txt ]; then
    echo ""
    echo "=== FAILED ==="
    cat /tmp/deploy_logs/failed.txt
fi
