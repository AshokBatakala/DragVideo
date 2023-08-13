# running draggan

# visualizer_auto.py
# - cuda device setting
# - pretrained_stylegan_weights path -> gloabl_state()

# viz/renderer.py
# - landmark points resolution error fixes

# visualizer_experiment.ipynb
import os
# import pickle
# DragGAN_dir = "/home/bean/DragVideo/DragGAN"
# os.chdir(DragGAN_dir)

from auto_drag import do_drag
from auto_drag import modify_landmarks

def run_dragvideo(Experiment_path,N_STEPS=100,CHECKPOINT_PATH=None):
    """
    CHECKPOINT_PATH: tuned_SG_pkl_path
    """

    landmarks_dir =  os.path.join(Experiment_path,'landmarks')

    def get_arguments(name):
        landmarks_path =os.path.join(landmarks_dir,str(name)+".pkl")# f"/home/bean/DragVideo/Data_store/data/PTI_results/landmarks/{name}.pkl"
        return {
            'w_load_path': os.path.join(Experiment_path,'latents','barcelona','PTI') + f"/{name}/0.pt",
            'stylegan2_wieghts_path' : CHECKPOINT_PATH,

            'points' : modify_landmarks(landmarks_path),
            'N_STEPS': N_STEPS,
            'save_path': os.path.join(Experiment_path,'after_drag')+f"/{name}.png",
        }
        
    latents_dir = os.path.join(Experiment_path,'latents') + "/barcelona/PTI"

    #  run draggan on based on latents availability

    temp = os.listdir(latents_dir)
    temp.sort()
    names = [i.split('.')[0] for i in temp]
    # names

    for name in names:
        args = get_arguments(name)
        do_drag(**args)
        
        

if __name__ == "__main__":
    
    #take input from user
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--Experiment_path", type=str, help="Experiment_path")
    parser.add_argument("--N_STEPS", type=int, help="N_STEPS")
    parser.add_argument("--CHECKPOINT_PATH", type=str, help="CHECKPOINT_PATH")
    args = parser.parse_args()
    
    Experiment_path = args.Experiment_path
    N_STEPS = args.N_STEPS
    CHECKPOINT_PATH = args.CHECKPOINT_PATH
    
    run_dragvideo(Experiment_path,N_STEPS=N_STEPS,CHECKPOINT_PATH=CHECKPOINT_PATH)
    print("Done!")
    
    
    