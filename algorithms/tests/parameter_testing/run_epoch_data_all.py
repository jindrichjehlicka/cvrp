import subprocess
import sys

scripts = ['./ga_sa/epoch_data.py', './genetic/epoch_data.py', './simulated_annealing/epoch_data.py',
           './tabu_search/epoch_data.py']

for script in scripts:
    try:
        print(f"Running {script}...")
        # Run the script file
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while running {script}: {e}")

print("All scripts have been processed.")
