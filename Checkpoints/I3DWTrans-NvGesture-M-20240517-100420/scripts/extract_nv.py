import cv2
import os

def extract_frames(video_path, output_root):
    for class_folder in os.listdir(video_path):
        class_folder_path = os.path.join(video_path, class_folder)
        if os.path.isdir(class_folder_path):
            for subject_folder in os.listdir(class_folder_path):
                subject_folder_path = os.path.join(class_folder_path, subject_folder)
                if os.path.isdir(subject_folder_path):
                    video_file_path = os.path.join(subject_folder_path, 'sk_color.avi')
                    if os.path.exists(video_file_path):
                        output_dir = os.path.join(output_root, class_folder, subject_folder)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        extract_frames_from_video(video_file_path, output_dir)

    print("Extraction complete.")

def extract_frames_from_video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"{count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        count += 1

        if count % 100 == 0:
            print(f"Extracted {count}/{frame_count} frames.")

    cap.release()

# Example usage:


def read_log(log_file_path):
    with open(log_file_path, 'r') as log_file:
        video_length = sum(1 for line in log_file)
    return video_length
    
import cv2
import os
import random

def get_frame_count(video_path):
    """ Returns the number of frames in a video file. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

import os
import random

import os
import random

import os
import random

import os
import random

def create_train_valid_split(data_root, train_ratio=0.7):
    # Define directories
    video_data_path = os.path.join(data_root, 'Video_data')
    classes = sorted([d for d in os.listdir(video_data_path) if os.path.isdir(os.path.join(video_data_path, d)) and "class" in d])
    
    train_list = []
    valid_list = []
    for class_id, class_folder in enumerate(classes, start=0):
        class_path = os.path.join(video_data_path, class_folder)
        subjects = sorted(os.listdir(class_path))
        
        for subject in subjects:
            subject_path = os.path.join(class_path, subject)
            video_files = [f for f in os.listdir(subject_path) if f.endswith('.avi') and "sk_color.avi" in f]
            
            # Ensure only one sk_color.avi file per subject
            if video_files:
                video = video_files[0]
                frame_count = get_frame_count(os.path.join(subject_path, video))
                # Create a cleaned up filename entry without 'sk_color.avi'
                # Remove 'sk_color.avi' and trim the trailing slash
                cleaned_path = f"{class_folder}/{subject}/{video.replace('sk_color.avi', '')}".strip()
                if cleaned_path.endswith('/'):
                    cleaned_path = cleaned_path[:-1]
                entry = f"{cleaned_path} {frame_count} {class_id}"
                
                # Split into train or validation set
                if random.random() < train_ratio:
                    train_list.append(entry)
                else:
                    valid_list.append(entry)

    # Shuffle train and validation lists
    
    # Save to file
    with open(os.path.join(data_root, 'train.txt'), 'w') as f:
        for item in train_list:
            f.write("%s\n" % item)
    
    with open(os.path.join(data_root, 'valid.txt'), 'w') as f:
        for item in valid_list:
            f.write("%s\n" % item)

    return train_list, valid_list


# Example usage:
# create_train_valid_lists('/path/to/nvGesture/Video_data', '/path/to/output', train_ratio=0.7)

    
file_path = '/home/minjae/ws/data/nvGesture'
create_train_valid_split(file_path, train_ratio=0.9)



# video_path = "/home/minjae/ws/data/nvGesture/Video_data"
# output_root = "/home/minjae/ws/data/nvGesture/Video_data/rgb"
# extract_frames(video_path, output_root)

