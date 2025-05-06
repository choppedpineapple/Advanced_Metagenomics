#!/bin/bash

# Define cleanup function (modify as needed)
cleanup() {
    echo "Performing cleanup tasks..."
    # Add your cleanup commands here (e.g., remove temporary files, kill subprocesses)
}

# Signal handler function
handler() {
    local sig_name=$1
    local sig_num
    sig_num=$(kill -l "$sig_name" 2>/dev/null)  # Get signal number from name

    # Ignore subsequent signals while handling
    trap '' SIGINT SIGTERM

    # Prompt user in a loop to handle potential read failures
    local response=""
    while [[ ! "$response" =~ ^[YyNn]$ ]]; do
        read -p "Interrupted by ${sig_name}. Perform cleanup before exiting? [y/N] " -n 1 -r
        echo  # move to new line
        response="${REPLY:-N}"  # Default to 'N' if empty
    done

    if [[ "$response" =~ ^[Yy]$ ]]; then
        cleanup
    fi

    # Restore default signal handling and exit with proper code
    trap - SIGINT SIGTERM
    exit $((128 + sig_num))
}

# Set traps for signals
trap 'handler SIGINT' SIGINT
trap 'handler SIGTERM' SIGTERM

# Example main script workflow
echo "Script started with PID $$"
echo "Running main process..."

# Simulate work - replace with your actual workflow
for i in {1..10}; do
    echo "Processing item $i..."
    sleep 1
done

# Normal exit if not interrupted
echo "Script completed successfully"
exit 0
