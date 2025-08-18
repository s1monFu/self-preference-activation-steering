#!/bin/bash

NUMBER=15
# List of names to iterate through
NAMES=(
    "gpt4"
    #"llama3.1-70b-instruct-fp8"
    #"deepseek-r1-0528" reasoning ahh
    #"qwen3-32b-fp8" reasoning ahh
    #"lfm-40b" no log probs, will think about how to patch
    #"hermes3-8b" already did
    "llama3.2-3b-instruct"
    # "llama3.1-405b-instruct-fp8"
    "deepseek-v3-0324"
    #"llama3.1-8b-instruct"
    "llama3.3-70b-instruct-fp8"
    "hermes3-405b"
    # "llama3.2-11b-vision-instruct"
    #"qwen25-coder-32b-instruct" i bet its going to try to reason
    "llama-4-maverick-17b-128e-instruct-fp8"
    #"deepseek-llama3.3-70b"
    #"deepseek-r1-671b" reasoning model
    # "llama3.1-nemotron-70b-instruct-fp8"
    #"lfm-7b" no log probs?
)

# First, generate summaries for all models
echo "--- Generating summaries for all models ---"
for NAME in "${NAMES[@]}"; do
    echo "Checking/Generating summaries for: $NAME"
    
    N_SUFFIX="_$NUMBER" # Since N is 50
    XSUM_FILE="summaries/xsum/xsum_train_${NAME}_responses${N_SUFFIX}.json"
    CNN_FILE="summaries/cnn/cnn_train_${NAME}_responses${N_SUFFIX}.json"

    # Check if summary files exist for both xsum and cnn
    if [ ! -f "$XSUM_FILE" ] || [ ! -f "$CNN_FILE" ]; then
        echo "Generating summaries for $NAME"
        python3 generate_summaries.py "$NAME" -N $NUMBER
        
        if [ $? -ne 0 ]; then
            echo "Error: generate_summaries.py failed for $NAME"
            continue
        fi
    else
        echo "Summaries already exist for $NAME, skipping"
    fi
done

# # Add all changes to git after all summaries are generated
# git add .
# git commit -m "Generated summaries for all models"
# git push

# Now run experiments once with all models as SOURCES
echo "--- Running experiments with all models ---"
SOURCES=$(IFS=,; echo "${NAMES[*]}")
echo "Executing: python3 experiments.py $SOURCES $NUMBER compare"
python3 experiments.py "$SOURCES" $NUMBER compare

if [ $? -ne 0 ]; then
    echo "Error: experiments.py failed"
    exit 1
fi

# # Final git commit for experiment results
# git add .
# git commit -m "Experiment results for all models"
# git push

echo "--- All operations completed ---"
