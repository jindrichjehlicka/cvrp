import subprocess
import sys

scripts = ['./ga_sa/epoch_data.py', './genetic/epoch_data.py', './simulated_annealing/epoch_data.py',
           './tabu_search/epoch_data.py']

repeat_count = 20

for script in scripts:
    for i in range(repeat_count):
        try:
            print(f"Running {script} (Iteration {i+1}/{repeat_count})...")
            # Run the script file
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script} (Iteration {i+1}): {e}")
        except Exception as e:
            print(f"An unexpected error occurred while running {script} (Iteration {i+1}): {e}")

print("We got all the epoch data!!!")
