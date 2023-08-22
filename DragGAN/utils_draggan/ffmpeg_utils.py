
#==============================================================================
# some useful commands for ffmpeg 
#==============================================================================
_ = f"""

for extracting frames from video
ffmpeg -i delete.mp4 -vf "fps=30" -start_number 0 -q:v 1 ./frames/%03d.png

make video of .png files of that folder

ffmpeg -framerate 30 -pattern_type glob -i './*.png' -c:v libx264 -pix_fmt yuv420p output_ffmpeg.mp4 -y

for side by side video

1.this is working in stylegan3 env
 ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4


2.throwing error ( i think because of audio channel)
ffmpeg -i left.mp4 -i right.mp4 -filter_complex \
"[0:v][1:v]hstack=inputs=2[v]; \
 [0:a][1:a]amerge[a]" \
-map "[v]" -map "[a]" -ac 2 output.mp4


"""


#==============================================================================
# ffmped class 
#==============================================================================
import os 
import subprocess

class ffmpeg:
    @staticmethod
    def extract_frames(video_path,
                       output_frames_dir,
                       n_digits_in_name=3,
                       ext='png',
                       fps=24):
        # ffmpeg -i delete.mp4 -vf "fps=24" -start_number 0 -q:v 1 ./frames/%03d.png
        command = f'ffmpeg -i {video_path} -vf "fps={fps}" -start_number 0 -q:v 1 {output_frames_dir}/%0{n_digits_in_name}d.{ext}'
        subprocess.call(command,shell=True)
        

    @staticmethod
    def make_video(images_dir,
                      ext='png',
                      video_name='output',
                      fps=24,
                      video_dir=None,
                      ):
        
        if video_dir is None:
            video_dir = images_dir
        video_path = os.path.join(video_dir,video_name+".mp4")
        
        command = f'ffmpeg -framerate {fps} -pattern_type glob -i "./*.{ext}" -c:v libx264 -pix_fmt yuv420p {video_path} -y'
        subprocess.call(command,shell=True,cwd=images_dir)      
        
    @staticmethod
    def hstack_videos(video1_path,
                         video2_path,
                         output_dir="./",
                         output_name="combined_s2s"
                            ):
        
        output_video_path = os.path.join(output_dir,output_name+".mp4")
        print(output_video_path)
        command = f'ffmpeg -i {video1_path} -i {video2_path} -filter_complex "[0:v] [1:v] hstack=inputs=2:shortest=1" {output_video_path} -y'
        subprocess.call(command,shell=True)
        
        
    @staticmethod
    def vstack_videos(video1_path,
                         video2_path,
                         output_dir="./",
                         output_name="combined_s2s"
                            ):
        
        output_video_path = os.path.join(output_dir,output_name+".mp4")
        print(output_video_path)
        command = f'ffmpeg -i {video1_path} -i {video2_path} -filter_complex "[0:v] [1:v] vstack=inputs=2:shortest=1" {output_video_path} -y'
        subprocess.call(command,shell=True)