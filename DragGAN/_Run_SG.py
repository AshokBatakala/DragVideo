# Author: Ashok B
# date: 2023/08/25

# to decode latent using stylegan2 or stylegan3
#-----------------------------------------------
print("use _Run_SG.lazy_Run instead of _Run_SG.Run  ")

import torch
import pickle

def load_G(sg_pkl):
    with open(sg_pkl, "rb") as f:
        data = pickle.load(f)
    key = "G_ema"
    _device = "cuda"
    
    if "stylegan2" in sg_pkl:
        from training.networks_stylegan2 import Generator
    elif "stylegan3" in sg_pkl:
        from training.networks_stylegan3 import Generator
    else:
        print("Error: name of the model should contain stylegan2 or stylegan3")
        
    net = Generator(*data[key].init_args, **data[key].init_kwargs)
    net.load_state_dict(data[key].state_dict())
    net.eval()
    net.to(_device)
    return net




def run_SG(
                      G,
                      ws,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 10,
        reg             = 0,
        feature_idx     = 5,
        r1              = 3,
        r2              = 12,
        random_seed     = 0,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        force_fp32      = False,
        layer_name      = None,
        sel_channels    = 3,
        base_channel    = 0,
        img_scale_db    = 0,
        img_normalize   = False,
        untransform     = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        _device = "cuda",
        **kwargs
    ):
        if ws.dim() == 2:
            ws = ws.unsqueeze(1).repeat(1,6,1)
        # Run synthesis network.
        label = torch.zeros([1, G.c_dim], device=_device)
        img, feat = G(ws, label,
                      truncation_psi=trunc_psi,
                      noise_mode=noise_mode,
                      input_is_w=True,
                      return_feature=True)

        h, w = G.img_resolution, G.img_resolution
        # return img
    
        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        
        return img.detach().cpu().numpy()

        
            
            
def plot(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    
    
def Run(sg_pkl,ws_path,_device="cuda",want_plot=False):
    G = load_G(sg_pkl).to(_device)
    #check if ws is .pt or .pkl
    if ws_path.endswith('.pt'):
        ws = torch.load(ws_path)
    else:
        ws = pickle.load(open(ws_path,'rb'))
        ws = torch.tensor(ws)
    ws = ws.to(_device)
    img = run_SG(G,ws)
    if want_plot:
        plot(img)
    return img


def lazy_Run(sg_pkl,ws_path,_device="cuda",want_plot=False):
    """ takes either a tensor or a file path as input for ws_path"""
    import numpy as np
    G = load_G(sg_pkl).to(_device)
    
    #check if ws is file path or tensor
    if isinstance(ws_path,str):
        #check if ws is .pt or .pkl
        if ws_path.endswith('.pt'):
            ws = torch.load(ws_path)
        else:
            ws = pickle.load(open(ws_path,'rb'))
            ws = torch.tensor(ws)
    else:
        #check if it is tensor or numpy array
        if isinstance(ws_path,np.ndarray):
            ws = torch.tensor(ws_path)
        elif isinstance(ws_path,torch.Tensor):
            ws = ws_path
        else:
            print('Error: ws_path should be either a file path, tensor or numpy array')
            return None
    
    ws = ws.to(_device)
    img = run_SG(G,ws)
    if want_plot:
        plot(img)
    return img