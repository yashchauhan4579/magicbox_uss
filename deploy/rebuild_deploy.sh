#!/bin/bash
set -e
echo "=== Building complete deploy package ==="

WORK=/userdata/linaro/head_deploy_final
rm -rf $WORK
mkdir -p $WORK/scripts $WORK/models $WORK/venv

# Copy pipeline script and model
cp /userdata/linaro/head_count/scripts/head_pipeline_v3.py $WORK/scripts/
cp /userdata/linaro/head_count/models/best_head_fp16_320.rknn $WORK/models/

# Copy stripped venv from the existing mini deploy
cp -a /userdata/linaro/head_deploy_mini/venv/* $WORK/venv/

# Add back missing deps from the original venv
SP_SRC=/userdata/linaro/projects/.venv/lib/python3.9/site-packages
SP_DST=$WORK/venv/lib/python3.9/site-packages
for pkg in ruamel _ruamel_yaml.cpython-39-aarch64-linux-gnu.so ruamel_yaml-0.17.40.dist-info \
           psutil psutil-5.9.5.dist-info \
           humanfriendly humanfriendly-10.0.dist-info \
           coloredlogs coloredlogs-15.0.1.dist-info \
           packaging packaging-24.2.dist-info; do
    if [ -e "$SP_SRC/$pkg" ]; then
        cp -a "$SP_SRC/$pkg" "$SP_DST/" 2>/dev/null && echo "  Added: $pkg"
    fi
done

# Fix venv paths
find $WORK/venv/bin -type f -exec sed -i 's|/userdata/linaro/projects/.venv|/userdata/linaro/head_deploy/venv|g' {} + 2>/dev/null || true

# Create service file
cat > $WORK/head-pipeline-v3.service <<'EOF'
[Unit]
Description=IRIS Head Detection Pipeline v3
After=network-online.target usscore.service ussstreamcontroller.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/usr/local/uss
Environment=PATH=/userdata/linaro/head_deploy/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/userdata/linaro/head_deploy/venv/bin/python3 /userdata/linaro/head_deploy/scripts/head_pipeline_v3.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create complete install script that also creates the config
cat > $WORK/install.sh <<'INST'
#!/bin/bash
set -e
DEPLOY_DIR=/userdata/linaro/head_deploy
echo "=== IRIS Head Pipeline Installer ==="

echo "1. Extracting to $DEPLOY_DIR ..."
mkdir -p $DEPLOY_DIR
# Copy everything from current dir (extracted tarball)
cp -r scripts models venv head-pipeline-v3.service $DEPLOY_DIR/

echo "2. Creating config..."
cat > /usr/local/uss/head-pipeline.json <<'CONF'
{
  "report_url": "http://10.100.0.37:9010/api/magicbox-crowd/ingest",
  "model_path": "/userdata/linaro/head_deploy/models/best_head_fp16_320.rknn",
  "input_size": 320,
  "cycle_interval_sec": 5,
  "conf_threshold": 0.35,
  "iou_threshold": 0.45
}
CONF

echo "3. Installing service..."
cp $DEPLOY_DIR/head-pipeline-v3.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable head-pipeline-v3
systemctl restart head-pipeline-v3

echo "4. Verifying..."
sleep 5
systemctl status head-pipeline-v3 --no-pager | head -10
echo ""
journalctl -u head-pipeline-v3 --no-pager -n 15
echo "=== INSTALL COMPLETE ==="
INST
chmod +x $WORK/install.sh

# Check sizes
echo ""
echo "=== Component sizes ==="
du -sh $WORK/scripts/ $WORK/models/ $WORK/venv/
echo "=== Total ==="
du -sh $WORK/

# Create tarball
echo "Creating tarball..."
cd $WORK
tar czf /userdata/linaro/head_deploy_final.tar.gz .
ls -lh /userdata/linaro/head_deploy_final.tar.gz

# Verify deps are in tarball
echo ""
echo "=== Verifying tarball contents ==="
tar tzf /userdata/linaro/head_deploy_final.tar.gz | grep -E 'ruamel/|psutil/' | head -3
tar tzf /userdata/linaro/head_deploy_final.tar.gz | grep install.sh

echo "=== DONE ==="
