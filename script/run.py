# example usage: python run 42

import subprocess
import sys

argument = sys.argv[1] # get argument from the command line, 42 in our example
number_of_jobs = 3
for job_number in range(number_of_jobs):

    # write your command as a list of words
    command = ["echo", str(argument), str(job_number)]

    # run command and store output that would go to the terminal
    process = subprocess.run(command, capture_output=True) # run your command

    # check that the process has run and print the output
    print(process.returncode) # 0 if completed
    print(process.stdout) # captured output
