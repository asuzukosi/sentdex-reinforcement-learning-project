import numpy as np
import cv2
import os
 
def generate_video_from_numpy_array(renders, height, width, channel=3, outfile_name="test.mp4", fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
    
    #Syntax: cv2.VideoWriter( filename, fourcc, fps, frameSize )
    video = cv2.VideoWriter(outfile_name, fourcc, float(fps), (width, height))
    
    for frame_count in range(len(renders)):
        video.write(renders[frame_count])
    
    video.release()
    