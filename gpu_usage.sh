#!/bin/bash

# GPU Usage Aggregation Script
# This script runs squeue to get GPU information and aggregates it per node.

# Function to display help
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help    Show this help message"
    echo "  --test        Run with mock data for testing"
    exit 0
}

# Function to parse and aggregate
aggregate_gpus() {
    local input_data="$1"
    declare -A node_gpus

    while IFS='|' read -r nodelist gres; do
        [[ -z "$nodelist" || "$nodelist" == "(null)" || "$nodelist" == "N/A" ]] && continue
        
        # Extract GPU count from GRES (e.g., gpu:2, gpu:tesla:4, gpu:0, etc.)
        count=0
        if [[ "$gres" =~ gpu:([0-9]+) ]]; then
            count="${BASH_REMATCH[1]}"
        elif [[ "$gres" =~ gpu:[^:]+:([0-9]+) ]]; then
            count="${BASH_REMATCH[1]}"
        fi

        # Expand nodelist (e.g., n[01-04] -> n01 n02 n03 n04)
        # Using scontrol show hostnames is the standard Slurm way
        if [[ "$nodelist" == *"["* ]]; then
            if command -v scontrol &> /dev/null; then
                expanded_nodes=$(scontrol show hostnames "$nodelist")
            else
                # Fallback for systems without scontrol or for testing without it
                expanded_nodes=$(echo "$nodelist" | sed -e 's/\[//' -e 's/\]//' -e 's/,/ /g')
            fi
        else
            expanded_nodes="$nodelist"
        fi

        for node in $expanded_nodes; do
            node_gpus["$node"]=$(( ${node_gpus["$node"]:-0} + count ))
        done
    done <<< "$input_data"

    # Print results
    echo "---------------------------"
    echo "Node            GPUs Used"
    echo "---------------------------"
    for node in "${!node_gpus[@]}"; do
        printf "%-15s %d\n" "$node" "${node_gpus[$node]}"
    done | sort
    echo "---------------------------"
}

# Check for flags
if [[ "$1" == "--test" ]]; then
    echo "Running with MOCK data..."
    MOCK_DATA="n01|gpu:2
n02|gpu:4
n[03-04]|gpu:1
n05|(null)
n01|gpu:0"
    # Mocking scontrol for the test case
    scontrol() {
        if [[ "$1" == "show" && "$2" == "hostnames" ]]; then
            if [[ "$3" == "n[03-04]" ]]; then
                echo -e "n03\nn04"
            else
                echo "$3"
            fi
        fi
    }
    # In bash, function is available at the current scope
    aggregate_gpus "$MOCK_DATA"
    exit 0
elif [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Standard execution
if ! command -v squeue &> /dev/null; then
    echo "Error: squeue command not found. Are you on a Slurm login node?"
    exit 1
fi

# Get NODELIST and GRES for RUNNING jobs only
# We use -t R for Running, -h to hide headers, and -o for simple formatted output with | delimiter
SQUEUE_OUTPUT=$(squeue -t R -h -o "%N|%b")
aggregate_gpus "$SQUEUE_OUTPUT"
