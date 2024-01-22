import subprocess
from os import listdir

# change these if you are using different directories
in_dir = "./scores"
out_dir = "./"
old_file_dir = "./old-scores"

print_results = False

# get tsv files from the last round (making sure to put them all in IN_DIR first) 
files = listdir(in_dir)
invoke_command = ["python", "score_final.py"]

for i, file_name in enumerate(files):
    # remove spaces and the word "final" from the file name
    new_file_name = file_name.replace(" ", "").replace("final", "")
    if new_file_name != file_name:
        subprocess.run(["mv", f"{in_dir}/{file_name}", f"{in_dir}/{new_file_name}"])
        files[i] = new_file_name
    # add individual dance score files to the list of arguments for the invoke command
    invoke_command.append(f"{in_dir}/{new_file_name}")

# output results to separate directory
invoke_command.extend(["-p"] if print_results else ["--output_dir", out_dir])

print(f"Running {invoke_command}")
print(f"Scoring {files}\n\n")
subprocess.run(invoke_command)
# NOTE: if you have a duplicate file name in OLD_FILE_DIR, then it will be overwritten by the file from this round
subprocess.run(f"mv {in_dir}/* {old_file_dir}", shell=True)
print("\n\n")
