PIDS=$(pgrep -f run.sh)

# Check if any PID was found
if [ -z "$PIDS" ]; then
  echo "No processes found for run.sh."
  exit 1
fi

# Iterate through each PID and kill the corresponding process
for PID in $PIDS; do
  echo "Terminating process with ID: $PID"
  kill -9 $PID

  # Check if the process was successfully killed
  if [ $? -eq 0 ]; then
    echo "Successfully terminated process with ID: $PID"
  else
    echo "Failed to terminate process with ID: $PID"
  fi
done
ps aux | grep 'vllm.entrypoints.api_server ' | grep -v 'vim' | awk '{print $2}' | xargs kill -9
echo "All processes for run.sh have been terminated."
