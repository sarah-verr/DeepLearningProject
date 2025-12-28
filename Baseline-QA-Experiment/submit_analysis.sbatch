#!/bin/bash
#SBATCH --job-name=llava_baseline_qa_all
#SBATCH --account=deep_learning
#SBATCH --output=logs/attn_%j.out
#SBATCH --error=logs/attn_%j.err
#SBATCH --time=24:00:00         # 4 hours (adjust based on dataset size)

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if a specific strategy is provided as argument
STRATEGY_ARG=""
if [ -n "$1" ]; then
    STRATEGY_ARG="$1"
fi

# Start timing
START_TIME=$(date +%s)
START_DATE=$(date)

if [ -z "$STRATEGY_ARG" ]; then
    STRATEGY_DESC="All Strategies per Image"
else
    STRATEGY_DESC="Strategy $STRATEGY_ARG Only"
fi

echo "========================================"
echo "  VLM Baseline QA Analysis Batch Job"
echo "  ($STRATEGY_DESC)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $START_DATE"
echo "========================================"

# Configuration
export HF_HOME="/work/scratch/$USER"
VENV_NAME="venv"
DATA_PATH="$HOME/DeepLearningProject/Synthetic-Data/vlm_levels"
SCRIPT_NAME="prompt_baseline_qa.py"  # Baseline QA script with all-strategies support

# Activate virtual environment
echo ">> Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Levels to process
LEVELS=(0)

# Track total images processed (each image processes all 4 strategies)
TOTAL_IMAGES=0
SUCCESSFUL_IMAGES=0
FAILED_IMAGES=0

# Process each level
for LEVEL in "${LEVELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing Level $LEVEL"
    echo "========================================"
    
    LEVEL_START=$(date +%s)
    
    IMAGES_DIR="$DATA_PATH/level_$LEVEL/images"
    ANN_DIR="$DATA_PATH/level_$LEVEL/ann"
    
    # Check if directories exist
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "Error: Images directory not found: $IMAGES_DIR"
        continue
    fi
    
    if [ ! -d "$ANN_DIR" ]; then
        echo "Error: Annotations directory not found: $ANN_DIR"
        continue
    fi
    
    # Count images
    NUM_IMAGES=$(ls "$IMAGES_DIR"/*.png 2>/dev/null | wc -l)
    echo "Found $NUM_IMAGES images in level $LEVEL"
    
    # Process each image (all strategies at once)
    COUNTER=0
    for IMAGE_PATH in "$IMAGES_DIR"/*.png; do
        # Extract image ID (filename without extension)
        IMAGE_FILE=$(basename "$IMAGE_PATH")
        IMAGE_ID="${IMAGE_FILE%.png}"
        
        # Check if annotation exists
        ANN_PATH="$ANN_DIR/${IMAGE_ID}.json"
        if [ ! -f "$ANN_PATH" ]; then
            echo "Warning: Annotation not found for $IMAGE_ID, skipping..."
            continue
        fi
        
        COUNTER=$((COUNTER + 1))
        TOTAL_IMAGES=$((TOTAL_IMAGES + 1))
        
        echo ""
        echo "[$COUNTER/$NUM_IMAGES] Processing: $IMAGE_ID (All Strategies)"
        echo "--------------------------------------------"
        
        # Time individual image processing
        IMAGE_START=$(date +%s)
        
        # Run baseline QA testing
        if [ -z "$STRATEGY_ARG" ]; then
            # Run all strategies
            python "$SCRIPT_NAME" \
                --level "$LEVEL" \
                --id "$IMAGE_ID" \
                --all-strategies
        else
            # Run specific strategy
            python "$SCRIPT_NAME" \
                --level "$LEVEL" \
                --id "$IMAGE_ID" \
                --strategy "$STRATEGY_ARG"
        fi
        
        IMAGE_END=$(date +%s)
        IMAGE_DURATION=$((IMAGE_END - IMAGE_START))
        
        # Check if successful
        if [ $? -eq 0 ]; then
            if [ -z "$STRATEGY_ARG" ]; then
                echo "✓ Successfully processed $IMAGE_ID (All Strategies) in ${IMAGE_DURATION}s"
            else
                echo "✓ Successfully processed $IMAGE_ID (Strategy $STRATEGY_ARG) in ${IMAGE_DURATION}s"
            fi
            SUCCESSFUL_IMAGES=$((SUCCESSFUL_IMAGES + 1))
        else
            if [ -z "$STRATEGY_ARG" ]; then
                echo "✗ Error processing $IMAGE_ID (All Strategies) (took ${IMAGE_DURATION}s)"
            else
                echo "✗ Error processing $IMAGE_ID (Strategy $STRATEGY_ARG) (took ${IMAGE_DURATION}s)"
            fi
            FAILED_IMAGES=$((FAILED_IMAGES + 1))
        fi
    done
    
    LEVEL_END=$(date +%s)
    LEVEL_DURATION=$((LEVEL_END - LEVEL_START))
    LEVEL_MINUTES=$((LEVEL_DURATION / 60))
    LEVEL_SECONDS=$((LEVEL_DURATION % 60))
    
    echo ""
    if [ -z "$STRATEGY_ARG" ]; then
        echo "Completed level $LEVEL: Processed $COUNTER images (all strategies) in ${LEVEL_MINUTES}m ${LEVEL_SECONDS}s"
    else
        echo "Completed level $LEVEL: Processed $COUNTER images (strategy $STRATEGY_ARG) in ${LEVEL_MINUTES}m ${LEVEL_SECONDS}s"
    fi
done

# Calculate total time
END_TIME=$(date +%s)
END_DATE=$(date)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "========================================"
echo "       Batch Job Complete!"
echo "========================================"
echo "Start time:      $START_DATE"
echo "End time:        $END_DATE"
echo "Total duration:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Statistics:"
if [ -z "$STRATEGY_ARG" ]; then
    echo "  Total images:      $TOTAL_IMAGES (each with all strategies)"
else
    echo "  Total images:      $TOTAL_IMAGES (each with strategy $STRATEGY_ARG)"
fi
echo "  Successful:        $SUCCESSFUL_IMAGES"
echo "  Failed:            $FAILED_IMAGES"
if [ $TOTAL_IMAGES -gt 0 ]; then
    AVG_TIME=$((TOTAL_DURATION / TOTAL_IMAGES))
    if [ -z "$STRATEGY_ARG" ]; then
        echo "  Avg time/image:    ${AVG_TIME}s (includes all strategies)"
    else
        echo "  Avg time/image:    ${AVG_TIME}s (strategy $STRATEGY_ARG only)"
    fi
fi
echo "========================================"