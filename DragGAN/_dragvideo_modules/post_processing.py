#usage 
#--------------------
#inside docker dravideo:v5
#(stylegan3) bean@pikachu:~/DragVideo/DragGAN$ python -m _dragvideo_modules.post_processing
#--------------------
from utils_draggan import ffmpeg_utils
from _PTI.utils import de_alignment 
import os 
from tqdm import tqdm
put_back_the_edited_image = de_alignment.put_back_the_edited_image

def paste_edited_faces_back(input_dir,
                            output_dir,
                            raw_dir = "raw",
                            edited_dir='after_drag',
                            save_dir='after_drag_pasted',
                            quad_path='quad_values',):
    """
    input_dir: raw,quad_values
    output_dir: after_drag_pasted,after_drag
    """
    
    raw_dir = os.path.join(input_dir,raw_dir)
    edited_dir = os.path.join(output_dir,edited_dir)
    quad_dir = os.path.join(input_dir,quad_path)
    save_dir = os.path.join(output_dir,save_dir)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # get names from edited_dir
    names = [name.split('.')[0] for name in os.listdir(edited_dir)]
    for name in tqdm(names):
        raw_image = os.path.join(raw_dir,name+'.png')
        edited_image = os.path.join(edited_dir,name+'.png')
        quad_path = os.path.join(quad_dir,name+'.pkl')
        save_path = os.path.join(save_dir,name+'.png')
        put_back_the_edited_image(raw_image,edited_image,quad_path,save_path=save_path)

def main(input_dir,
        output_dir,
         video_path,
            fps=24,
            ):
    """
    it pastes back the edited faces to the original video
    then, it combines the edited video and original video
    
    for both before and after drag
    
    dirs:
    - input_dir: raw,quad_values
    - output_dir: after_drag,before_drag should be there.
                before_drag_pasted,after_drag_pasted,videos are created
                
    """
    #check if after_drag_pasted,before_drag_pasted,video exists
    for dir_name in ['after_drag_pasted','before_drag_pasted','videos']:
        os.makedirs(os.path.join(output_dir,dir_name),exist_ok=True)

    paste_edited_faces_back(input_dir,output_dir,edited_dir="before_drag",save_dir="before_drag_pasted")
    paste_edited_faces_back(input_dir,output_dir,edited_dir="after_drag",save_dir="after_drag_pasted")

    # =============================================================================
    # using ffmpeg to make video
    # =============================================================================
    before_drag_pasted_dir = os.path.join(output_dir,'before_drag_pasted')
    after_drag_pasted_dir = os.path.join(output_dir,'after_drag_pasted')
    videos_dir = os.path.join(output_dir,'videos')


    ffmpeg_utils.ffmpeg.make_video(before_drag_pasted_dir,
                                video_name="pre_drag_full",
                                video_dir=videos_dir,
                                fps=fps
                                    )
    ffmpeg_utils.ffmpeg.make_video(after_drag_pasted_dir,
                                    video_name="post_drag_full",
                                        video_dir=videos_dir,
                                        fps=fps
                                        )

    # =============================================================================
    # hstack videos
    # =============================================================================

    video1= os.path.join(videos_dir,"pre_drag_full.mp4")
    video2= os.path.join(videos_dir,"post_drag_full.mp4")

    # pre_post_drag_full
    ffmpeg_utils.ffmpeg.hstack_videos(video1,
                                    video2,
                                    output_dir=videos_dir,
                                    output_name="hstack_pre_post_drag_full",
    )

    # original_post_drag_full
    ffmpeg_utils.ffmpeg.hstack_videos(video_path,
                                    video2,
                                    output_dir=videos_dir,
                                    output_name="hstack_original_post_drag_full",
    )

if __name__ == "__main__":
    
    base_dir = "/home/bean/"
    
    input_dir = f"{base_dir}DragVideo/Data_store/experiments/woman_waiting_input"
    output_dir = f"{base_dir}DragVideo/Data_store/outputs/woman_waiting_1"
    video_path = f"{base_dir}DragVideo/Data_store/raw_videos/woman_waiting.mp4"
        
    fps = 24
    
    main(input_dir,output_dir,video_path,fps=fps)