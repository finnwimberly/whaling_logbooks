# Jupyter Notebook Management Skills

This file provides instructions for Antigravity on how to manage the Jupyter Notebook server for this repository.

## 1. Starting Jupyter Notebook
To start the Jupyter Notebook using conda:
- First, discover the correct workspace directory if you are unsure of it (e.g., by checking your workspace metadata or running a command to find the repository root).
- Execute the following command using the `run_command` tool, setting `Cwd` to the discovered workspace directory:
  `~/miniconda3/condabin/conda run --no-capture-output -n ewhales jupyter notebook --ip 0.0.0.0 --port 8099 --no-browser`
- Ensure that you send it to the background so that it keeps running without blocking you.
- The command will be tracked as a background task. Take note of its Task ID.

## 2. Retrieving the URL
Once the notebook is started:
- Use the `run_command` tool to execute `hostname` to discover the current machine's hostname.
- Use the `view_file` tool to read the task log file of the background task you just created.
- You might need to use the `schedule` tool to wait a few seconds if the log is empty, as Jupyter takes a moment to start.
- Search the log output for the URL containing the access token, using the hostname you looked up (it will look something like `http://<discovered_hostname>:8099/tree?token=...`).
- Provide this URL directly to the user so they can access the notebook.

## 3. Shutting down Jupyter Notebook
To stop the notebook server:
- Use the `manage_task` tool with the action `list` to find the running task ID for the Jupyter Notebook.
- Use the `manage_task` tool with the action `kill` on the corresponding Task ID to shut it down.
