# Ashok added this file

print("loading _do_drag.py")
import torch
from ._visualizer_auto import DragVideo_Base
import os
from _dragpoint_utils import large_eyes,make_jaw_wider,mouth_wide,large_nose,nose_to_mouth,smile,up_eyebrows


def do_drag(w_load_path=None,
            stylegan2_wieghts_path=None,
            points = dict(),
            N_STEPS=50,
            save_path=None,
            save_before_drag_path = None,
            image_show_path = None,
            edited_latents_dir=None,
            verbose=False
    ):
    
    w_load = torch.load(w_load_path)
    drag_video = DragVideo(w_load=w_load,
                        stylegan2_wieghts_path=stylegan2_wieghts_path,
                        edited_latents_dir=edited_latents_dir,
                        verbose=verbose)
    
    feat = drag_video.run(N_STEPS=N_STEPS,points=points)
    
    # drag_video.global_state['images'].keys() ==>  ['image_orig', 'image_raw', 'image_show']
    if save_path is not None:
        image = drag_video.global_state['images']['image_raw']
        image.save(save_path)
    
    if  save_before_drag_path is not None:
        image = drag_video.global_state['images']['image_orig']
        image.save(save_before_drag_path)
    if image_show_path is not None:
        image = drag_video.global_state['images']['image_show']
        image.save(image_show_path)
        
    return drag_video.global_state['images']#['image_raw']



class DragVideo(DragVideo_Base):
    def __init__(self,
                #  w_load=None,
                #  stylegan2_wieghts_path=None,
                #  device = "cuda",
                # verbose=False,
                # use_border_mask = True,
                # border_mask_fraction = 0.1,
                inputs_dir = None,
                outputs_dir = None,
                image_size = 1024,
                **kwargs):
                 
        super().__init__(**kwargs)
        
        self.inputs_dir = inputs_dir
        self.image_size = image_size
        
        # folders to save outputs
        if os.path.isdir(outputs_dir):
            self.outputs_dir = outputs_dir
            for dir_name in ['after_drag','before_drag','image_show','edited_latents']:
                os.makedirs(os.path.join(self.outputs_dir,dir_name),exist_ok=True)
        else:
            raise Exception(f"outputs_dir {self.outputs_dir} does not exists")
        
    def run(self,N_STEPS=50,
            edit_mode="smile"):
        torch.cuda.empty_cache()
        latents,landmarks,names = self.get_available_latents_and_landmarks()
        zip_list = zip(latents,landmarks,names)
        for w_load_path,landmark_path,name in zip_list:
            points = self.get_drag_points(edit_mode,landmark_path,MAX_SIZE=self.image_size)
            self.drag_one_frame(w_load_path,points,N_STEPS=N_STEPS)
            self.save_outputs(name)
                
    def get_available_latents_and_landmarks(self):
        try:
            lantents_dir = os.path.join(self.inputs_dir,"latents","barcelona","PTI")
            landmarks_dir = os.path.join(self.inputs_dir,"landmarks")
        except:
            raise Exception("inputs_dir is not set properly")
        
        temp = os.listdir(lantents_dir)
        temp.sort()
        names = [i.split('.')[0] for i in temp]
        
        #check if landmarks are available for all latents
        for name in names:
            if not os.path.isfile(os.path.join(landmarks_dir,f"{name}.pkl")):
                print(f"landmarks for {name} is not available.. removing from list")
                names.remove(name)
        print(f"Total {len(names)} images to be processed...")
        
        landmarks = [os.path.join(landmarks_dir,f"{name}.pkl") for name in names]
        latents = [os.path.join(lantents_dir,f"{name}","0.pt") for name in names]
        return latents,landmarks,names
    @staticmethod
    def get_drag_points(edit_mode,
                             landmarks_path,
                             MAX_SIZE=1024):
        """
        create points,targets using landmarks and editing_function_name
        """
        options = {
            "large_eyes":large_eyes.large_eyes,
            "make_jaw_wider":make_jaw_wider.make_jaw_wider,
            "mouth_wide":mouth_wide.mouth_wide,
            "large_nose":large_nose.large_nose,
            "nose_to_mouth":nose_to_mouth.nose_to_mouth,
            "smile": smile.smile,
            "up_eyebrows": up_eyebrows.up_eyebrows,
        }
        print(f"editing mode is: {edit_mode}")
        func =  options[edit_mode]
        return func(landmarks_path,MAX_SIZE=MAX_SIZE)
    
    def save_outputs(self,frame):
        
        out = self.outputs_dir
        save_after_drag_path = os.path.join(out,'after_drag',f"{frame}.png")
        save_before_drag_path = os.path.join(out,'before_drag',f"{frame}.png")
        save_image_show_path = os.path.join(out,'image_show',f"{frame}.png")
        save_edited_latent_path = os.path.join(out,'edited_latents',f"{frame}.pt")
        
        if save_after_drag_path is not None:
            image = self.global_state['images']['image_raw']
            image.save(save_after_drag_path)
        if  save_before_drag_path is not None:
            image = self.global_state['images']['image_orig']
            image.save(save_before_drag_path)
        if save_image_show_path is not None:
            image = self.global_state['images']['image_show']
            image.save(save_image_show_path)
        if save_edited_latent_path is not None:
            torch.save(self.outputs['edited_latent'],save_edited_latent_path)
            
    
        
    