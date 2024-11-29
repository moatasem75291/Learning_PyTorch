import subprocess
import os


def run_cmd_commands():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    command = 'cmd.exe /k "cd /d {} && conda activate cuda_env && jupyter notebook"'.format(
        current_dir
    )
    subprocess.Popen(command, shell=True)


if __name__ == "__main__":
    run_cmd_commands()
