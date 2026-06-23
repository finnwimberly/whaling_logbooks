# Python Execution Skills

This file provides instructions for Antigravity on how to execute Python code in this repository.

## Conda Environment Requirement
- **IMPORTANT**: All Python commands in this repository MUST be run using the conda environment defined in `conda-environment.yml` (the `ewhales` environment).
- **DO NOT** use the system python or simply run `python script.py` directly.

## Running Python with Conda
To run Python scripts or tests using conda:
1. Always use the conda binary located at `~/miniconda3/condabin/conda`.
2. First, check if the `ewhales` environment exists by running `~/miniconda3/condabin/conda env list`.
3. If the `ewhales` environment does not exist, install it by running `~/miniconda3/condabin/conda env create -f conda-environment.yml`.
4. Use the `conda run` command, specifying the environment with `-n ewhales`.
5. To execute a Python script, the command format is:
   `~/miniconda3/condabin/conda run -n ewhales python <path_to_script.py>`
6. To execute tests using pytest, the command format is:
   `~/miniconda3/condabin/conda run -n ewhales pytest <path_to_test_file.py>`
7. Use the `run_command` tool in the workspace directory (`/home/alexander.laties/ewhales-v1/data-pipeline`) to execute these commands.

## Managing Dependencies
- **CRITICAL RULE**: Whenever you install or uninstall a dependency in the `ewhales` environment, you MUST immediately update the `conda-environment.yml` file.
- Update the file by exporting the environment state and piping the output directly to the file:
  `~/miniconda3/condabin/conda env export -n ewhales > conda-environment.yml`
- **IMPORTANT**: After exporting, always strip the `prefix:` line from the bottom of the `conda-environment.yml` file (e.g., using a code editing tool or `sed -i '/^prefix: /d' conda-environment.yml`), so the environment file remains portable.
