# Ashok added this file 
# (modified from visualized_drag_gradio.py)
# __

# check " on_click_start " function for drag optimization



import os
import pickle
import os.path as osp
from argparse import ArgumentParser
from functools import partial

import gradio as gr
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import dnnlib
from gradio_utils import (ImageMask, draw_mask_on_image, draw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from viz.renderer import Renderer, add_watermark_np



parser = ArgumentParser()
parser.add_argument('--share', action='store_true',default='True')
parser.add_argument('--cache-dir', type=str, default='./checkpoints') # path of stylegan2 models weights
parser.add_argument(
    "--listen",
    action="store_true",
    help="launch gradio with 0.0.0.0 as server name, allowing to respond to network requests",
)
# args = parser.parse_args() # this causes error when run from jupyter notebook
args, unknown = parser.parse_known_args()

cache_dir = './checkpoints'#args.cache_dir

device = 'cuda'


def reverse_point_pairs(points):
    new_points = []
    for p in points:
        new_points.append([p[1], p[0]])
    return new_points


def clear_state(global_state, target=None):
    """Clear target history state from global_state
    If target is not defined, points and mask will be both removed.
    1. set global_state['points'] as empty dict
    2. set global_state['mask'] as full-one mask.
    """
    if target is None:
        target = ['point', 'mask']
    if not isinstance(target, list):
        target = [target]
    if 'point' in target:
        global_state['points'] = dict()
        print('Clear Points State!')
    if 'mask' in target:
        image_raw = global_state["images"]["image_raw"]
        global_state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                       dtype=np.uint8)
        print('Clear mask State!')

    return global_state


def init_images(global_state,w_load=None,stylegan2_wieghts_path=None):
    """
    * w_load: latent (w_pivot) from PTI model
    
    This function is called only ones with Gradio App is started.
    0. pre-process global_state, unpack value from global_state of need
    1. Re-init renderer
    2. run `renderer._render_drag_impl` with `is_drag=False` to generate
       new image
    3. Assign images to global state and re-generate mask
    """

    if stylegan2_wieghts_path is None:
        stylegan2_wieghts_path = valid_checkpoints_dict[global_state['pretrained_weight']]


    if isinstance(global_state, gr.State):
        state = global_state.value
    else:
        state = global_state

    state['renderer'].init_network(
        state['generator_params'],  # res
        # valid_checkpoints_dict[state['pretrained_weight']],  # pkl # ------------- "styleagan checkpoint path"
        stylegan2_wieghts_path,
        state['params']['seed'],  # w0_seed,
        # None,  # w_load
        w_load,
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr,
    )

    state['renderer']._render_drag_impl(state['generator_params'],
                                        is_drag=False,              # why is it calling with is_drag=False; does it intialization anything?
                                        to_pil=True)

    init_image = state['generator_params'].image
    state['images']['image_orig'] = init_image
    state['images']['image_raw'] = init_image
    state['images']['image_show'] = Image.fromarray(
        add_watermark_np(np.array(init_image)))
    state['mask'] = np.ones((init_image.size[1], init_image.size[0]),
                            dtype=np.uint8)
    return global_state


def update_image_draw(image, points, mask, show_mask, global_state=None):

    image_draw = draw_points_on_image(image, points)
    if show_mask and mask is not None and not (mask == 0).all() and not (
            mask == 1).all():
        image_draw = draw_mask_on_image(image_draw, mask)

    image_draw = Image.fromarray(add_watermark_np(np.array(image_draw)))
    if global_state is not None:
        global_state['images']['image_show'] = image_draw
    return image_draw


def preprocess_mask_info(global_state, image):
    """Function to handle mask information.
    1. last_mask is None: Do not need to change mask, return mask
    2. last_mask is not None:
        2.1 global_state is remove_mask:
        2.2 global_state is add_mask:
    """
    if isinstance(image, dict):
        last_mask = get_valid_mask(image['mask'])
    else:
        last_mask = None
    mask = global_state['mask']

    # mask in global state is a placeholder with all 1.
    if (mask == 1).all():
        mask = last_mask

    # last_mask = global_state['last_mask']
    editing_mode = global_state['editing_state']

    if last_mask is None:
        return global_state

    if editing_mode == 'remove_mask':
        updated_mask = np.clip(mask - last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do remove.')
    elif editing_mode == 'add_mask':
        updated_mask = np.clip(mask + last_mask, 0, 1)
        print(f'Last editing_state is {editing_mode}, do add.')
    else:
        updated_mask = mask
        print(f'Last editing_state is {editing_mode}, '
              'do nothing to mask.')

    global_state['mask'] = updated_mask
    # global_state['last_mask'] = None  # clear buffer
    return global_state


# def on_click_start(global_state,
#                    N_STEPS = 10,
#                     points=dict(),
#                     verbose=False):#, image):

#     # example for global_state['points'] ; it is only used for updating image_draw
#     # {0: {'start': [3, 4], 'target': [249, 214]},
#     # 1: {'start': [3, 243], 'target': [506, 239]}}

#     # Source: [[251, 360], [249, 380]]
#     # Target: [[251, 362], [251, 366]]

#     # global_state['points'] = {
#     #                         0: {'start': [251, 360], 'target': [251, 362]},
#     #                         1: {'start': [249, 380], 'target': [251, 366]},
#     #                         }
#     global_state['points'] = points
    

#     p_in_pixels = []
#     t_in_pixels = []
#     valid_points = []

#     # # handle of start drag in mask editing mode
#     # # global_state = preprocess_mask_info(global_state )#, image) 
#     # #Ashok# here only image['mask'] is used. other info is not used.


#     # Transform the points into torch tensors
#     for key_point, point in global_state["points"].items():
#         try:
#             p_start = point.get("start_temp", point["start"])
#             p_end = point["target"]

#             if p_start is None or p_end is None:
#                 continue

#         except KeyError:
#             continue

#         p_in_pixels.append(p_start)
#         t_in_pixels.append(p_end)
#         valid_points.append(key_point)

#     mask = torch.tensor(global_state['mask']).float()
#     drag_mask = 1 - mask

#     renderer: Renderer = global_state["renderer"]
#     global_state['temporal_params']['stop'] = False
#     global_state['editing_state'] = 'running'

#     # reverse points order
#     p_to_opt = reverse_point_pairs(p_in_pixels)
#     t_to_opt = reverse_point_pairs(t_in_pixels)


#     if verbose:
#         print('Running with:')
#         print(f'    Source: {p_in_pixels}')
#         print(f'    Target: {t_in_pixels}')
        
#     step_idx = 0

#     feature_map = None

#     # while True: # i changed this to for loop
#     # for _ in range(N_STEPS):
#     for _ in tqdm(range(N_STEPS)):
#         if verbose:
#             print(f'p_to_opt: {p_to_opt}')
#             print(f't_to_opt: {t_to_opt}')
#         # print(f'drag_mask: {drag_mask}')
#         if global_state["temporal_params"]["stop"]:
#             break

#         # do drage here!
#         feature_map = renderer._render_drag_impl(
#             global_state['generator_params'], # res? .. does it have ws in it?
#             p_to_opt,  # point
#             t_to_opt,  # target
#             drag_mask,  # mask,
#             global_state['params']['motion_lambda'],  # lambda_mask
#             reg=0,
#             feature_idx=5,  # NOTE: do not support change for now
#             r1=global_state['params']['r1_in_pixels'],  # r1
#             r2=global_state['params']['r2_in_pixels'],  # r2
#             # random_seed     = 0,
#             # noise_mode      = 'const',
#             trunc_psi=global_state['params']['trunc_psi'],
#             # force_fp32      = False,
#             # layer_name      = None,
#             # sel_channels    = 3,
#             # base_channel    = 0,
#             # img_scale_db    = 0,
#             # img_normalize   = False,
#             # untransform     = False,
#             is_drag=True,
#             to_pil=True)

#         if step_idx % global_state['draw_interval'] == 0:
#             if verbose:
#                 print('Current Source:')
#             for key_point, p_i, t_i in zip(valid_points, p_to_opt,
#                                             t_to_opt):
#                 global_state["points"][key_point]["start_temp"] = [
#                     p_i[1],
#                     p_i[0],
#                 ]
#                 global_state["points"][key_point]["target"] = [
#                     t_i[1],
#                     t_i[0],
#                 ]
#                 start_temp = global_state["points"][key_point][
#                     "start_temp"]
#                 if verbose:
#                     print(f'    {start_temp}')

#             image_result = global_state['generator_params']['image']
#             image_draw = update_image_draw(
#                 image_result,
#                 global_state['points'],
#                 global_state['mask'],
#                 global_state['show_mask'],
#                 global_state,
#             )
#             global_state['images']['image_raw'] = image_result
            
#             # store the edited latent 
#             edited_latent = global_state['generator_params'].w
#             pkl_name = 

#         # increate step
#         step_idx += 1

#     image_result = global_state['generator_params']['image']
#     global_state['images']['image_raw'] = image_result
#     image_draw = update_image_draw(image_result,
#                                     global_state['points'],
#                                     global_state['mask'],
#                                     global_state['show_mask'],
#                                     global_state)


#     global_state['editing_state'] = 'add_points'

#     return feature_map


class DragVideo:
    def __init__(self,
                    w_load=None,
                    stylegan2_wieghts_path=None,
                    pretrained_model_name=None,
                    edited_latents_dir=None,
                    verbose=False):

        # # make the directory for edited latents
        if not os.path.exists(edited_latents_dir):
            os.makedirs(edited_latents_dir)
        self.edited_latents_dir = edited_latents_dir
        
        self.previous_feature_map = None

        self.latents_dir = None
        self.points_targets_dir = None
        self.output_dir = None
        self.cache_dir = None
        self.verbose = verbose
        
        self.process_samples = []

        print('intiating global state....')
        self.global_state = {#gr.State({
                            "images": {
                                # image_orig: the original image, change with seed/model is changed
                                # image_raw: image with mask and points, change durning optimization
                                # image_show: image showed on screen
                            },
                            "temporal_params": {
                                # stop
                            },
                            'mask':
                            None,  # mask for visualization, 1 for editing and 0 for unchange
                            'last_mask': None,  # last edited mask
                            'show_mask': True,  # add button
                            "generator_params": dnnlib.EasyDict(),
                            "params": {
                                "seed": 1,
                                "motion_lambda": 20,
                                "r1_in_pixels": 3,
                                "r2_in_pixels": 12,
                                "magnitude_direction_in_pixels": 1.0,
                                "latent_space": "w+", # --- debug
                                "trunc_psi": 0.7,
                                "trunc_cutoff": None,
                                "lr": 0.001,
                            },
                            "device": device,
                            "draw_interval": 1,
                            "renderer": Renderer(disable_timing=True),
                            "points": {},
                            "curr_point": None,
                            "curr_type_point": "start",
                            'editing_state': 'add_points',
                            'pretrained_weight': pretrained_model_name,#'stylegan2-ffhq-512x512' # model weights pkl in cache_dir
                        # })
                        }
    
        print('calling init_images......')
        # init image
        self.global_state = init_images(self.global_state  ,w_load=w_load,stylegan2_wieghts_path=stylegan2_wieghts_path)



    def run(self,N_STEPS = 10,points = None):
        self.previous_feature_map =  self.on_click_start(self.global_state,
                N_STEPS = N_STEPS,
                points = points,
                verbose=self.verbose
                 )
        return self.previous_feature_map,self.global_state['images']['image_show']
    
    def drag_one_frame(self):
        pass
        
    def on_click_start(self,global_state,
                    N_STEPS = 10,
                        points=dict(),
                        verbose=False):#, image):

        # example for global_state['points'] ; it is only used for updating image_draw
        # {0: {'start': [3, 4], 'target': [249, 214]},
        # 1: {'start': [3, 243], 'target': [506, 239]}}

        # Source: [[251, 360], [249, 380]]
        # Target: [[251, 362], [251, 366]]

        # global_state['points'] = {
        #                         0: {'start': [251, 360], 'target': [251, 362]},
        #                         1: {'start': [249, 380], 'target': [251, 366]},
        #                         }
        global_state['points'] = points
        

        p_in_pixels = []
        t_in_pixels = []
        valid_points = []

        # # handle of start drag in mask editing mode
        # # global_state = preprocess_mask_info(global_state )#, image) 
        # #Ashok# here only image['mask'] is used. other info is not used.


        # Transform the points into torch tensors
        for key_point, point in global_state["points"].items():
            try:
                p_start = point.get("start_temp", point["start"])
                p_end = point["target"]

                if p_start is None or p_end is None:
                    continue

            except KeyError:
                continue

            p_in_pixels.append(p_start)
            t_in_pixels.append(p_end)
            valid_points.append(key_point)

        mask = torch.tensor(global_state['mask']).float()
        drag_mask = 1 - mask

        renderer: Renderer = global_state["renderer"]
        global_state['temporal_params']['stop'] = False
        global_state['editing_state'] = 'running'

        # reverse points order
        p_to_opt = reverse_point_pairs(p_in_pixels)
        t_to_opt = reverse_point_pairs(t_in_pixels)


        if verbose:
            print('Running with:')
            print(f'    Source: {p_in_pixels}')
            print(f'    Target: {t_in_pixels}')
            
        step_idx = 0

        feature_map = None

        # while True: # i changed this to for loop
        # for _ in range(N_STEPS):
        for _ in tqdm(range(N_STEPS)):
            if verbose:
                print(f'p_to_opt: {p_to_opt}')
                print(f't_to_opt: {t_to_opt}')
            # print(f'drag_mask: {drag_mask}')
            if global_state["temporal_params"]["stop"]:
                break

            # do drage here!
            feature_map = renderer._render_drag_impl(
                global_state['generator_params'], # res? .. does it have ws in it?
                p_to_opt,  # point
                t_to_opt,  # target
                drag_mask,  # mask,
                global_state['params']['motion_lambda'],  # lambda_mask
                reg=0,
                feature_idx=5,  # NOTE: do not support change for now
                r1=global_state['params']['r1_in_pixels'],  # r1
                r2=global_state['params']['r2_in_pixels'],  # r2
                # random_seed     = 0,
                # noise_mode      = 'const',
                trunc_psi=global_state['params']['trunc_psi'],
                # force_fp32      = False,
                # layer_name      = None,
                # sel_channels    = 3,
                # base_channel    = 0,
                # img_scale_db    = 0,
                # img_normalize   = False,
                # untransform     = False,
                is_drag=True,
                to_pil=True)

            if step_idx % global_state['draw_interval'] == 0:
                if verbose:
                    print('Current Source:')
                for key_point, p_i, t_i in zip(valid_points, p_to_opt,
                                                t_to_opt):
                    global_state["points"][key_point]["start_temp"] = [
                        p_i[1],
                        p_i[0],
                    ]
                    global_state["points"][key_point]["target"] = [
                        t_i[1],
                        t_i[0],
                    ]
                    start_temp = global_state["points"][key_point][
                        "start_temp"]
                    if verbose:
                        print(f'    {start_temp}')

                image_result = global_state['generator_params']['image']
                image_draw = update_image_draw(
                    image_result,
                    global_state['points'],
                    global_state['mask'],
                    global_state['show_mask'],
                    global_state,
                )
                global_state['images']['image_raw'] = image_result
                
                # ------------store the edited latent ----------------
                edited_latent = global_state['generator_params'].w
                pkl_name = os.path.join(self.edited_latents_dir,f'{step_idx}.pkl')
                with open(pkl_name,'wb') as f:
                    pickle.dump(edited_latent,f)

            # increate step
            step_idx += 1

        image_result = global_state['generator_params']['image']
        global_state['images']['image_raw'] = image_result
        image_draw = update_image_draw(image_result,
                                        global_state['points'],
                                        global_state['mask'],
                                        global_state['show_mask'],
                                        global_state)


        global_state['editing_state'] = 'add_points'

        return feature_map









