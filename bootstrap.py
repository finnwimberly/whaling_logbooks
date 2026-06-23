import os
import sys
import shutil
import argparse
import datetime
import re
import subprocess
import urllib.request

def prompt_user(question, default="y"):
    if not sys.stdout.isatty():
        print(f"{question} [Non-interactive, defaulting to '{default}']")
        return default.lower() in ['y', 'yes']
    
    valid = {"y": True, "n": False, "yes": True, "no": False}
    prompt = " [Y/n] " if default == "y" else " [y/N] "
    
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def check_and_install_conda():
    conda_path = shutil.which("conda")
    if conda_path:
        print(f"Conda found at: {conda_path}")
        return conda_path

    print("Conda binary not found in PATH.")
    if prompt_user("Would you like to auto-install conda via Miniforge to your home directory?"):
        print("Installing Miniforge...")
        miniforge_url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
        installer_path = os.path.join(os.path.expanduser("~"), "miniforge_installer.sh")
        install_dir = os.path.join(os.path.expanduser("~"), "miniforge3")
        
        try:
            print(f"Downloading Miniforge from {miniforge_url}...")
            urllib.request.urlretrieve(miniforge_url, installer_path)
            
            print(f"Running Miniforge installer (installing to {install_dir})...")
            subprocess.run(["bash", installer_path, "-b", "-p", install_dir], check=True)
            
            print("Initializing conda for bash...")
            subprocess.run([os.path.join(install_dir, "bin", "conda"), "init", "bash"], check=True)
            
            print("Miniforge installed successfully.")
            os.remove(installer_path)
            print("Please restart your shell or run 'source ~/.bashrc' after this script finishes to use conda.")
            
            return os.path.join(install_dir, "bin", "conda")
        except Exception as e:
            print(f"Failed to install conda: {e}")
            if os.path.exists(installer_path):
                os.remove(installer_path)
            return None
    else:
        print("Skipping conda installation.")
        return None

def setup_conda_env(conda_executable):
    if not os.path.exists("conda-environment.yml"):
        print("conda-environment.yml not found, skipping environment creation.")
        return
        
    if prompt_user("Would you like to create/update the conda environment from conda-environment.yml?"):
        print("Creating/updating conda environment...")
        try:
            subprocess.run([conda_executable, "env", "create", "-f", "conda-environment.yml"], check=True)
            print("Conda environment setup complete.")
        except subprocess.CalledProcessError:
            print("Failed to create conda environment (it might already exist).")
            if prompt_user("Would you like to try updating the existing environment instead?"):
                try:
                    subprocess.run([conda_executable, "env", "update", "-f", "conda-environment.yml"], check=True)
                    print("Conda environment update complete.")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to update conda environment: {e}")
    else:
        print("Skipping conda environment setup.")

def main():
    parser = argparse.ArgumentParser(description="Bootstrap a new data pipeline export directory.")
    parser.add_argument('csv_path', type=str, help="Path to the exported CSV file from WordPress")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: The provided CSV path '{args.csv_path}' does not exist.")
        sys.exit(1)

    today = datetime.date.today()
    new_dir_name = f"{today.strftime('%Y%m%d')}_export"

    if os.path.exists(new_dir_name):
        print(f"Directory {new_dir_name} already exists. Aborting.")
        sys.exit(1)

    # Find the most recent export folder
    export_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and re.match(r'^\d{8}_export$', d)]
    if not export_dirs:
        print("Error: No existing YYYYMMDD_export directories found to copy from.")
        sys.exit(1)

    # Sort by the YYYYMMDD part
    export_dirs.sort(reverse=True)
    most_recent_export = export_dirs[0]

    print(f"Found most recent export directory: {most_recent_export}")
    print(f"Creating new export directory: {new_dir_name}")

    # Define directories to exclude files from
    exclude_dirs = {'csv_files', 'pkl_files', 'output_txt_files'}

    # List of all directories specified in "Repository Structure"
    structure_dirs = [
        "csv_files",
        "pkl_files",
        "figures",
        "manuscript_figures",
        "meta_figs/single_voyages",
        "meta_figs/combined_voyages",
        "newsletter_figures",
        "output_txt_files",
        "permanent_txt_files",
        "utils"
    ]

    # Create the new base directory and the structure
    os.makedirs(new_dir_name)
    for d in structure_dirs:
        os.makedirs(os.path.join(new_dir_name, d), exist_ok=True)

    # Walk through the most recent export directory and copy files
    for root, dirs, files in os.walk(most_recent_export):
        # Calculate relative path from the most recent export dir
        rel_path = os.path.relpath(root, most_recent_export)
        
        # Determine if current directory is one of the excluded ones (or a subdirectory of them)
        parts = rel_path.split(os.sep)
        in_excluded_dir = parts[0] in exclude_dirs

        # Create corresponding directory in the new structure (if not already created)
        target_root = os.path.join(new_dir_name, rel_path) if rel_path != '.' else new_dir_name
        os.makedirs(target_root, exist_ok=True)

        for f in files:
            source_file = os.path.join(root, f)
            target_file = os.path.join(target_root, f)

            # Copy file if it is not inside an excluded directory
            if not in_excluded_dir:
                shutil.copy2(source_file, target_file)

    # Copy the new CSV file
    new_csv_filename = f"logentries-export-{today.strftime('%Y-%m-%d')}.csv"
    target_csv_path = os.path.join(new_dir_name, "csv_files", new_csv_filename)
    
    print(f"Copying {args.csv_path} to {target_csv_path}")
    shutil.copy2(args.csv_path, target_csv_path)

    print("Checking conda setup...")
    conda_executable = check_and_install_conda()
    if conda_executable:
        setup_conda_env(conda_executable)

    print("Bootstrap complete!")

if __name__ == '__main__':
    main()
