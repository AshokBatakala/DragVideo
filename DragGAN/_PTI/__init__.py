import os 

cwd = os.getcwd()
file_path = os.path.realpath(__file__)
file_dir = os.path.dirname(file_path)
os.chdir(file_dir)

#==============================================================================
# make imports here 

from .utils.align_data import pre_process_images
from .scripts.run_pti import run_PTI

from .run import load_generators,export_updated_pickle
from .configs import paths_config


def init_experiment_dir(Experiment_name,Experiment_base_path,shell_script_path = file_dir):
        
    import subprocess
    file_full_path = os.path.join(shell_script_path,'init_datadirs.sh')
    subprocess.call(['bash', file_full_path, Experiment_name, Experiment_base_path])



#==============================================================================
# change back to original directory
os.chdir(cwd)

