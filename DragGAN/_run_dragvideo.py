# running draggan

# visualizer_auto.py
# - cuda device setting
# - pretrained_stylegan_weights path -> gloabl_state()

# viz/renderer.py
# - landmark points resolution error fixes

# visualizer_experiment.ipynb
import os

from _do_drag import do_drag
from _dragpoint_utils import large_eyes,make_jaw_wider,mouth_wide,large_nose,nose_to_mouth

def run_dragvideo(Experiment_path,
                  N_STEPS=100,
                  CHECKPOINT_PATH=None,
                  MAX_SIZE=1024,
                  editing_function_name="large_eyes",
                  verbose=False):
    """
    one frames at a time to do_drag() function    
    CHECKPOINT_PATH: tuned_SG_pkl_path
    
    """
    # editing function
    editing_func_dict = {
        "large_eyes":large_eyes.large_eyes,
        "make_jaw_wider":make_jaw_wider.make_jaw_wider,
        "mouth_wide":mouth_wide.mouth_wide,
        "large_nose":large_nose.large_nose,
        "nose_to_mouth":nose_to_mouth.nose_to_mouth,
    }
    EDITING_FUNC = editing_func_dict[editing_function_name]
    landmarks_dir =  os.path.join(Experiment_path,'landmarks')
    
    # get arguments
    def get_arguments(name):
        landmarks_path =os.path.join(landmarks_dir,str(name)+".pkl")# f"/home/bean/DragVideo/Data_store/data/PTI_results/landmarks/{name}.pkl"
        return {
            'w_load_path': os.path.join(Experiment_path,'latents','barcelona','PTI') + f"/{name}/0.pt",
            'stylegan2_wieghts_path' : CHECKPOINT_PATH,
            # 'points': make_jaw_wider.make_jaw_wider(landmarks_path,MAX_SIZE=MAX_SIZE),
            'points': EDITING_FUNC(landmarks_path,MAX_SIZE=MAX_SIZE),
            'N_STEPS': N_STEPS,
            'save_path': os.path.join(Experiment_path,'after_drag')+f"/{name}.png",
            'save_before_drag_path': os.path.join(Experiment_path,'before_drag')+f"/{name}.png",
            'image_show_path': os.path.join(Experiment_path,'image_show')+f"/{name}.png",
            'edited_latents_dir': os.path.join(Experiment_path,'latents','edited',str(name)),
        }
        
    latents_dir = os.path.join(Experiment_path,'latents') + "/barcelona/PTI"

    #  run draggan on based on latents availability
    temp = os.listdir(latents_dir)
    temp.sort()
    names = [i.split('.')[0] for i in temp]

    for name in names:
        args = get_arguments(name)
        do_drag(**args,verbose=verbose)
        
        

if __name__ == "__main__":
    
    #take input from user
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Experiment_path", type=str, help="Experiment_path")
    parser.add_argument("--N_STEPS", type=int, help="N_STEPS")
    parser.add_argument("--CHECKPOINT_PATH", type=str, help="CHECKPOINT_PATH")
    parser.add_argument("--MAX_SIZE", type=int, help="MAX_SIZE",default=1024)
    parser.add_argument("--editing_function_name", type=str, help="editing_function_name",default="large_eyes")
    parser.add_argument("--verbose", type=bool, help="verbose",default=False)
    args = parser.parse_args()
    
    Experiment_path = args.Experiment_path
    N_STEPS = args.N_STEPS
    CHECKPOINT_PATH = args.CHECKPOINT_PATH

    print("editing_function_name:",args.editing_function_name)
    run_dragvideo(Experiment_path,
                  N_STEPS=N_STEPS,
                  CHECKPOINT_PATH=CHECKPOINT_PATH,
                  MAX_SIZE=args.MAX_SIZE,
                  editing_function_name=args.editing_function_name,
                  verbose=args.verbose)
    
    print("Done!")
    
    
    