import glob

import cv2
import os

# Directory containing PNG images
image_dir = "/raid/data/users/adarc/registration/forked-remote/experiments/short_n_classes=3_supervised=False_MI_IXI_cuda0"
# sada

# Output video file


# Frame rate (3 frames per second)
frame_rate = 6

# Get a list of image files in the directory
n_image_files = len(glob.glob(os.path.join(image_dir, "*.png")))
image_files = [os.path.join(image_dir, f"output_epoch={i}.png") for i in range(n_image_files)]  # Load images from 0 to 10

# Get the dimensions of the first image to set up the video writer
first_image = cv2.imread(image_files[0])
height, width, layers = first_image.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = os.path.join(image_dir, f"until_epoch{n_image_files}_{image_dir.split('/')[-1]}.mp4")
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Iterate through the image files and add them to the video
for image_file in image_files:
    frame = cv2.imread(image_file)
    video_writer.write(frame)

# Release the VideoWriter
video_writer.release()

print(f"Video created: {output_video}")
