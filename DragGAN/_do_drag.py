# Ashok added this file

import torch
from _visualizer_auto import DragVideo

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

