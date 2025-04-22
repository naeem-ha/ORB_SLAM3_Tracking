import cv2
import os
import glob

def generate_video_from_images(images_folder, output_video_path='output.mp4', fps=30):
    # Get all image files and sort them
    image_files = sorted(glob.glob(os.path.join(images_folder, '*.png')) + 
                         glob.glob(os.path.join(images_folder, '*.jpg')) + 
                         glob.glob(os.path.join(images_folder, '*.jpeg')))
    
    if not image_files:
        raise ValueError("No image files found in the provided folder.")

    # Read the first image to get frame size
    frame = cv2.imread(image_files[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))  # Ensure all frames are the same size
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage:
generate_video_from_images('/home/hamza-naeem/Documents/ORB_SLAM3/MH_01_easy/mav0/cam0/data')
