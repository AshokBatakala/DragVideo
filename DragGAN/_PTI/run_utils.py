# ========================================
#            UTILITY FUNCTIONS
# ========================================

import os 

def add_dummy_config(filename,text,ROOT_PATH = '.',mode = "w"):
    file_path = os.path.join(ROOT_PATH, filename)
    with open(file_path, mode) as f:
        f.write(text)
        

def generate_text(**kwargs):
    text = "" 
    for key, value in kwargs.items():
        if type(value) == str:
            value = f"'{value}'"
        text += f"{key}= {value}\n"
    return text

def add_dummy_config_from_dict(filename,dict_,ROOT_PATH = '.',mode = "w"):
    text = generate_text(**dict_)
    add_dummy_config(filename,text,ROOT_PATH = ROOT_PATH,mode = mode)
    
    
# ========================================
def import_module_from_path( path,name='_module'):
    # ex: path = "path/to/module.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_experiment_dir(Experiment_name,Experiment_base_path,shell_script_path = "/home/bean/DragVideo/DragGAN/_PTI/"):
        
    import subprocess
    subprocess.call(['bash', shell_script_path, Experiment_name, Experiment_base_path])




