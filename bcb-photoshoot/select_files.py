import csv
import subprocess

files = []

selects = "./selects.csv"
input_dir = ""
output_dir = ""

with open(selects, "r", newline = "") as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        files.extend(row)

for file in files:
    subprocess.run(f"cp {input_dir}/{file}.CR3 {output_dir}", shell=True)
