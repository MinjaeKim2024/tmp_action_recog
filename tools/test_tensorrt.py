import cv2
import torch
from torchvision import transforms
import torch
import argparse
import sys
import os
sys.path.append(os.path.join('..', os.path.abspath(os.path.join(os.getcwd()))) )

from utils import *
from lib import *
from config import Config

from util_for_tensorrt import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Load Congfile.')
parser.add_argument('--eval_only', action='store_true', help='Eval only. True or False?')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--nprocs', type=int, default=1)

parser.add_argument('--save_grid_image', action='store_true', help='Save samples?')
parser.add_argument('--save_output', action='store_true', help='Save logits?')
parser.add_argument('--demo_dir', type=str, default='./demo', help='The dir for save all the demo')
parser.add_argument('--resume', type=str, default='', help='resume model path.')

parser.add_argument('--distill-lamdb', type=float, default=0.0, help='initial distillation loss weight')

parser.add_argument('--drop_path_prob', type=float, default=0.5, help='drop path probability')
parser.add_argument('--save', type=str, default='Checkpoints', help='experiment dir')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--epoch', type=int, default=50, help='random seed')
parser.add_argument('--phase', type=str, default='valid', help='random seed')
parser.add_argument('--weight', type=str, default='/home/minjae/ws/MotionRGBD/model.engine', help='random seed')

args = parser.parse_args()
args = Config(args)

np.random.seed(123)
if "Jester" in args.config:
    
    num_classes = 27
    g_dict = {
    0: "Drumming Fingers", 
    1: "Sliding Two Fingers Right", 
    2: "Sliding Two Fingers Down", 
    3: "Pulling Two Fingers In", 
    4: "Sliding Two Fingers Up", 
    5: "Zooming Out With Two Fingers",
    6: "Pulling Hand In", 
    7: "Thumb Up", 
    8: "Swiping Right", 
    9: "Zooming In With Two Fingers",
    10: "Stop Sign", 
    11: "Doing other things", 
    12: "Swiping Down", 
    13: "No gesture", 
    14: "Thumb Down",
    15: "Rolling Hand Forward", 
    16: "Pushing Hand Away", 
    17: "Zooming Out With Full Hand", 
    18: "Shaking Hand",
    19: "Turning Hand Counterclockwise", 
    20: "Zooming In With Full Hand", 
    21: "Rolling Hand Backward",
    22: "Turning Hand Clockwise", 
    23: "Swiping Left", 
    24: "Sliding Two Fingers Left", 
    25: "Pushing Two Fingers Away", 
    26: "Swiping Up"
}
else:
    g_dict = {0:"move hand left", 1: "move hand right", 
          2:"move hand up", 3: "move hand down", 
          4:"move two fingers left", 5: "move two fingers right",
          6: "move two fingers up", 7: "move two fingers down",
          8: "click index finger", 9: "call someone",
          10: "open hand", 11: "shaking hand",
          12: "show index finger", 13: "show two fingers",
          14: "show three fingers", 15: "push hand up",
          16: "push hand down", 17: "push hand out",
          18: "pull hand in", 19: "rotate fingers cw",
          20: "rotate fingers ccw", 21: "push two fingers away",
          22: "close hand two times", 23: "thumb up",
          24: "okay sign"}
    num_classes = 25

print(args)
engine, context, inputs_, outputs, bindings, stream = initialize(args.weight)

# Dummy input data for testing
input_data_a = np.random.rand(1, 3, 64, 224, 224).astype(np.float32)
input_data_b = np.random.rand(1, 1, 64, 224, 224).astype(np.float32)

# Set input shapes if dynamic
input_binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
context.set_input_shape(input_binding_names[0], input_data_a.shape)
context.set_input_shape(input_binding_names[1], input_data_b.shape)



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

sample_size = args.sample_size
import time
cls_ = "nothing"
status = "nothing"
with torch.no_grad():
    while True:
        frame_list = []
        idx = -1
        pre_time = time.time()
        tot_time = 0
        for dx in range(64):
            status = "Gathering frames..."
            if dx == 63:
                status = "Recognition..."
            ret, frame = cap.read()
            put_text(frame, cls_, status)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) == ord('q'):
                break
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # idx +=1
            # if idx not in sl:
            #     continue
            
            tmp_time = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            frame_list.append(transform_compose(frame).view(3, sample_size, sample_size, 1))
            
            process_time = time.time() - tmp_time
            tot_time += process_time
        print("preprocess time:", tot_time)
        recognition_start = time.time()
        cl, skgmaparr = image_propose(frame_list)
        inputs, heatmap = cl.permute(0, 3, 1, 2), skgmaparr
        
        inputs, heatmap = map(lambda x: x.cuda(0, non_blocking=True), [inputs, heatmap])
        inputs, heatmap = inputs.unsqueeze(0), heatmap.unsqueeze(0)
        inputs_np = inputs.cpu().numpy()
        heatmap_np = heatmap.cpu().numpy()
        preprocess_input(inputs_, inputs_np, heatmap_np)

        # Run inference
        do_inference(context, bindings, inputs_, outputs, stream)

        # Reshape and return the output
        output_binding_name = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]
        output_shape = context.get_tensor_shape(output_binding_name)
        output_data = np.reshape(outputs[0][0], output_shape)
        # Print the output
        # i wanna get argmax of output data
        result = np.argmax(output_data)
        print("Inference output:", result)
        
        
        if result == 12:
            gesture = "nothing"
        else:
            gesture = g_dict[result]
        recognition_end = time.time()
        print("recognition time:", recognition_end - recognition_start)
        # print("max value:", torch.max(logits).item())
        # print(gesture,torch.argmax(logits).item())
        
        cls_ = gesture

        

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()

