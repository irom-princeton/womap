import cv2
import os
import numpy as np
from natsort import natsorted
from src.utils.gdino import compute_reward
from PIL import Image

def export_video(parent_dir, run_name, output_dir, fullres=True, img_resize_shape=None):
    image_dir = f"{parent_dir}/{run_name}/images/" 
    output_video = f"{output_dir}/{run_name}.mp4"
    predicted_confidences = np.load(f"{parent_dir}/{run_name}/predicted_confidence.npy")

    frame_rate = 10  # Frames per second

    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    images = natsorted(images)  # Sort the images in natural order (00000, 00001, etc.)

    if not images:
        raise ValueError("No PNG images found in the directory!")

    # get frame size
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if fullres:
        video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    else:
        video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, img_resize_shape)
    
    # text size
    if fullres:
        text_scale = width / 1280 #* 4
    else:
        text_scale =  img_resize_shape[0] / 1280 * 4

    text_freq = 5
    confidence_value_pred = predicted_confidences[0]

    for i, image in enumerate(images):
        image_path = os.path.join(image_dir, image)
        frame = cv2.imread(image_path)
        if not fullres:
            frame = cv2.resize(frame, img_resize_shape)
        # update values
        if i % text_freq == 0:
            confidence_value_pred = predicted_confidences[i]
            raw_image = Image.open(image_path)
            if img_resize_shape != None:
                raw_image = raw_image.resize(img_resize_shape)
            confidence_value_true, bbox_true = compute_reward(raw_image=raw_image)
        # annotate frame
        x_offset = int(10 * text_scale)
        y_offset = int(30 * text_scale)
        y_increment = int(50 * text_scale)
        cv2.putText(
            frame, 
            f"i={i}", 
            (x_offset, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 1
            )
        cv2.putText(
            frame, 
            f"P={confidence_value_pred:.4f}", 
            (x_offset, y_offset + y_increment), 
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), 1
            )
        cv2.putText(
            frame, 
            f"T={confidence_value_true:.4f}", 
            (x_offset, y_offset + y_increment * 2), 
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), 1
            )
        # convert to unit8
        frame = frame.astype(np.uint8)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"*** Video saved as {output_video} ***")