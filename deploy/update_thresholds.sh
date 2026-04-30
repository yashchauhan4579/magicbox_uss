#!/bin/bash
# Push updated config + patch DEFAULTS + restart service on all devices
CONFIG="/tmp/_head_pipeline_config.json"
LOG="/tmp/threshold_update.log"
> $LOG

update_device() {
    IP=$1
    SS="sshpass -p wiredleap12**"
    SO="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"

    # SCP config
    if ! $SS scp $SO "$CONFIG" linaro@$IP:/home/linaro/head_pipeline.json </dev/null >/dev/null 2>&1; then
        echo "$IP — FAIL (SCP)" | tee -a $LOG
        return 1
    fi

    # Move config + patch DEFAULTS + restart
    $SS ssh $SO linaro@$IP "echo 'wiredleap12**' | sudo -S bash -c '
        mv /home/linaro/head_pipeline.json /usr/local/uss/head-pipeline.json 2>/dev/null
        for S in /userdata/linaro/head_deploy/scripts/head_pipeline_v3.py /userdata/linaro/head_count/scripts/head_pipeline_v3.py; do
            if [ -f \$S ]; then
                sed -i \"s/\\\"conf_threshold\\\": 0.2/\\\"conf_threshold\\\": 0.3/\" \$S
                sed -i \"s/\\\"iou_threshold\\\": 0.4/\\\"iou_threshold\\\": 0.45/\" \$S
            fi
        done
        systemctl restart head-pipeline-v3
    '" </dev/null >/dev/null 2>&1

    echo "$IP — OK" | tee -a $LOG
}

export -f update_device
export CONFIG LOG

TOTAL=$(wc -l < /tmp/magicbox_devices.txt)
echo "=== Updating $TOTAL devices (conf=0.3, iou=0.45) ==="
echo "Started: $(date)"

cat /tmp/magicbox_devices.txt | xargs -P 15 -I {} bash -c 'update_device "$@"' _ {}

OK=$(grep -c "OK" $LOG)
FAIL=$(grep -c "FAIL" $LOG)
echo ""
echo "=== DONE — $(date) ==="
echo "Success: $OK"
echo "Failed: $FAIL"
if [ $FAIL -gt 0 ]; then
    echo "--- Failed ---"
    grep FAIL $LOG
fi
