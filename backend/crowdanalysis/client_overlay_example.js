/**
 * Client-side JavaScript example for rendering JSON overlay data on live video
 * This demonstrates how to recreate the crowd density heatmap overlay using JSON data
 * instead of a base64-encoded image for better performance with live streams.
 */

class CrowdOverlayRenderer {
    constructor(videoElement, canvasElement) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.overlayData = null;
        this.jetColormap = this.createJetColormap();
        
        // Resize canvas to match video
        this.resizeCanvas();
        
        // Listen for video resize events
        this.video.addEventListener('loadedmetadata', () => this.resizeCanvas());
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        this.canvas.width = this.video.videoWidth || this.video.clientWidth;
        this.canvas.height = this.video.videoHeight || this.video.clientHeight;
    }
    
    /**
     * Create a jet colormap similar to matplotlib's 'jet'
     * Returns an array of RGBA values for 256 levels
     */
    createJetColormap() {
        const colormap = [];
        for (let i = 0; i < 256; i++) {
            const t = i / 255.0;
            let r, g, b;
            
            if (t < 0.125) {
                r = 0;
                g = 0;
                b = 0.5 + 0.5 * (t / 0.125);
            } else if (t < 0.375) {
                r = 0;
                g = (t - 0.125) / 0.25;
                b = 1;
            } else if (t < 0.625) {
                r = (t - 0.375) / 0.25;
                g = 1;
                b = 1 - (t - 0.375) / 0.25;
            } else if (t < 0.875) {
                r = 1;
                g = 1 - (t - 0.625) / 0.25;
                b = 0;
            } else {
                r = 1 - 0.5 * (t - 0.875) / 0.125;
                g = 0;
                b = 0;
            }
            
            colormap.push([
                Math.round(r * 255),
                Math.round(g * 255),
                Math.round(b * 255),
                255
            ]);
        }
        return colormap;
    }
    
    /**
     * Update overlay with new JSON data from API
     */
    updateOverlay(overlayData) {
        this.overlayData = overlayData;
        this.renderOverlay();
    }
    
    /**
     * Main rendering function
     */
    renderOverlay() {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.overlayData || !this.overlayData.has_data) {
            return;
        }
        
        if (this.overlayData.format === 'sparse') {
            this.renderSparseOverlay();
        } else {
            this.renderDenseOverlay();
        }
    }
    
    /**
     * Render sparse format overlay data
     */
    renderSparseOverlay() {
        const { sparse_data, shape, downsample_factor, quantization_levels } = this.overlayData.density_map;
        const { alpha } = this.overlayData.display_settings;
        const { height: targetHeight, width: targetWidth } = this.overlayData.target_dimensions;
        
        if (sparse_data.coordinates.length === 0) {
            return;
        }
        
        // Calculate scale factors
        const scaleX = this.canvas.width / targetWidth;
        const scaleY = this.canvas.height / targetHeight;
        const cellWidth = (this.canvas.width / shape.width) * downsample_factor;
        const cellHeight = (this.canvas.height / shape.height) * downsample_factor;
        
        // Set global alpha for blending
        this.ctx.globalAlpha = alpha;
        
        // Render each density point
        for (let i = 0; i < sparse_data.coordinates.length; i++) {
            const [y, x] = sparse_data.coordinates[i];
            const value = sparse_data.values[i];
            
            // Normalize value to 0-255 range
            const normalizedValue = Math.round((value / (quantization_levels - 1)) * 255);
            const color = this.jetColormap[normalizedValue];
            
            // Calculate position on canvas
            const canvasX = x * cellWidth;
            const canvasY = y * cellHeight;
            
            // Draw density cell
            this.ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`;
            this.ctx.fillRect(canvasX, canvasY, cellWidth, cellHeight);
        }
        
        // Reset global alpha
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Render dense format overlay data
     */
    renderDenseOverlay() {
        const { data, shape, downsample_factor } = this.overlayData.density_map;
        const { alpha } = this.overlayData.display_settings;
        
        // Create ImageData for efficient rendering
        const imageData = this.ctx.createImageData(shape.width, shape.height);
        const pixels = imageData.data;
        
        // Fill pixel data
        for (let y = 0; y < shape.height; y++) {
            for (let x = 0; x < shape.width; x++) {
                const dataIndex = y * shape.width + x;
                const pixelIndex = dataIndex * 4;
                const value = data[dataIndex];
                const color = this.jetColormap[value];
                
                pixels[pixelIndex] = color[0];     // R
                pixels[pixelIndex + 1] = color[1]; // G
                pixels[pixelIndex + 2] = color[2]; // B
                pixels[pixelIndex + 3] = Math.round(color[3] * alpha); // A
            }
        }
        
        // Create temporary canvas for scaling
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = shape.width;
        tempCanvas.height = shape.height;
        
        // Put image data on temporary canvas
        tempCtx.putImageData(imageData, 0, 0);
        
        // Scale and draw to main canvas
        this.ctx.globalAlpha = alpha;
        this.ctx.drawImage(
            tempCanvas,
            0, 0, shape.width, shape.height,
            0, 0, this.canvas.width, this.canvas.height
        );
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Toggle overlay visibility
     */
    toggleOverlay(visible) {
        this.canvas.style.display = visible ? 'block' : 'none';
    }
    
    /**
     * Update overlay opacity
     */
    setOpacity(opacity) {
        if (this.overlayData) {
            this.overlayData.display_settings.alpha = opacity;
            this.renderOverlay();
        }
    }
}

/**
 * Example usage with WebSocket for real-time updates
 */
class RealTimeCrowdMonitor {
    constructor(videoElement, canvasElement, websocketUrl) {
        this.renderer = new CrowdOverlayRenderer(videoElement, canvasElement);
        this.ws = new WebSocket(websocketUrl);
        this.setupWebSocket();
    }
    
    setupWebSocket() {
        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                // Check if this is crowd counting data with overlay
                if (data.metricType === 'crowd_count' && data.overlayData) {
                    this.renderer.updateOverlay(data.overlayData);
                    
                    // Update UI with metrics
                    this.updateMetricsDisplay(data);
                }
            } catch (error) {
                console.error('Error processing WebSocket message:', error);
            }
        };
        
        this.ws.onopen = () => {
            console.log('Connected to crowd monitoring WebSocket');
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected from crowd monitoring WebSocket');
            // Implement reconnection logic here
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    updateMetricsDisplay(data) {
        // Update crowd count display
        const countElement = document.getElementById('crowd-count');
        if (countElement) {
            countElement.textContent = data.value_numeric;
        }
        
        // Update density level
        const densityElement = document.getElementById('density-level');
        if (densityElement) {
            densityElement.textContent = data.metadata.density_level;
            densityElement.className = `density-${data.metadata.density_level}`;
        }
        
        // Update hotspot information if available
        if (data.metadata.num_hotspots !== undefined) {
            const hotspotsElement = document.getElementById('hotspots-count');
            if (hotspotsElement) {
                hotspotsElement.textContent = data.metadata.num_hotspots;
            }
        }
        
        // Update critical areas percentage
        if (data.metadata.critical_density_area_pct !== undefined) {
            const criticalElement = document.getElementById('critical-area-pct');
            if (criticalElement) {
                criticalElement.textContent = `${data.metadata.critical_density_area_pct.toFixed(1)}%`;
            }
        }
    }
}

/**
 * Initialize the real-time crowd monitoring system
 */
function initializeCrowdMonitoring(videoElementId, canvasElementId, websocketUrl) {
    const video = document.getElementById(videoElementId);
    const canvas = document.getElementById(canvasElementId);
    
    if (!video || !canvas) {
        console.error('Video or canvas element not found');
        return null;
    }
    
    // Position canvas over video
    canvas.style.position = 'absolute';
    canvas.style.top = video.offsetTop + 'px';
    canvas.style.left = video.offsetLeft + 'px';
    canvas.style.pointerEvents = 'none'; // Allow clicks to pass through
    
    return new RealTimeCrowdMonitor(video, canvas, websocketUrl);
}

// Example HTML structure:
/*
<div class="video-container" style="position: relative;">
    <video id="crowd-video" width="800" height="600" autoplay>
        <source src="rtsp://your-camera-stream" type="application/x-mpegURL">
    </video>
    <canvas id="overlay-canvas"></canvas>
    
    <div class="metrics-panel">
        <div>Crowd Count: <span id="crowd-count">0</span></div>
        <div>Density Level: <span id="density-level">low</span></div>
        <div>Hotspots: <span id="hotspots-count">0</span></div>
        <div>Critical Areas: <span id="critical-area-pct">0%</span></div>
        
        <button onclick="monitor.renderer.toggleOverlay(false)">Hide Overlay</button>
        <button onclick="monitor.renderer.toggleOverlay(true)">Show Overlay</button>
        <input type="range" min="0" max="1" step="0.1" value="0.5" 
               oninput="monitor.renderer.setOpacity(this.value)">
    </div>
</div>

<script>
// Initialize the monitoring system
const monitor = initializeCrowdMonitoring(
    'crowd-video', 
    'overlay-canvas', 
    'ws://localhost:8080/crowd-updates'
);
</script>
*/ 