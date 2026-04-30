#!/bin/bash

# Convert TS files to MP4 format
# Source directory
SOURCE_DIR="/var/lib/mediamtx/recordings"
# Output directory
OUTPUT_DIR="mp4"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# List of files to convert
files=(
    "cam1-2025-06-28_22-26-13.ts"
    "cam1-2025-06-28_22-27-14.ts"
    "cam1-2025-06-28_22-28-17.ts"
    "cam1-2025-06-28_22-29-18.ts"
    "cam1-2025-06-28_22-30-19.ts"
    "cam1-2025-06-28_22-31-20.ts"
)

# Convert each file
for file in "${files[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        # Extract filename without extension
        filename=$(basename "$file" .ts)
        
        echo "Converting $file to $filename.mp4..."
        
        # Use ffmpeg to convert TS to MP4
        ffmpeg -i "$SOURCE_DIR/$file" \
               -c:v libx264 \
               -c:a aac \
               -preset medium \
               -crf 23 \
               "$OUTPUT_DIR/$filename.mp4"
        
        if [ $? -eq 0 ]; then
            echo "Successfully converted $file to $filename.mp4"
        else
            echo "Error converting $file"
        fi
    else
        echo "File $file not found in $SOURCE_DIR"
    fi
done

echo "Conversion complete!"
