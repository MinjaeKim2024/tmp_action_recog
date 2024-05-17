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
from PIL import Image
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
args = parser.parse_args()
args = Config(args)

np.random.seed(123)
if "Jester" in args.config:
    
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

class Normalize(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x
transform_compose = transforms.Compose([Normalize(), transforms.ToTensor()])

def transform(img):
    sample_size = args.sample_size
    resize = eval(args.resize)
    crop_rect = (16,16, 240, 240) # left, top, left + crop_size, top + crop_size

    img = np.asarray(img)
    if img.shape[-1] != 3:
        print("Image shape error!")
        exit()
    img = Image.fromarray(img) # 640 480
    img = img.resize(resize) # 256 256
    img = img.crop(crop_rect) # 224 224
    return np.array(img.resize((sample_size, sample_size)))

def tensor_arr_rp(arr):
    l = len(arr)
    statics = []
    _w = args.w
    def tensor_rankpooling(video_arr, lamb=1.):
        N = len(video_arr)
        # re = torch.zeros(video_arr[0].size(0), 1, video_arr[0].size(2), video_arr[0].size(3)).cuda()
        re = torch.zeros(video_arr[0].size())
        for a, b in zip(video_arr, [float(i) * 2 - N - 1 for i in range(1, N + 1)]):
            re += a * b
        re = F.relu(re) * lamb
        re -= torch.min(re)
        re = re / torch.max(re) if torch.max(re) != 0 else re / (torch.max(re) + 0.00001)

        re = transforms.Grayscale(1)(re.squeeze())
        # Static Attention
        static = torch.where(re > torch.mean(re), re, torch.full_like(re, 0))
        static = np.asarray(static.squeeze())
        # static = cv2.morphologyEx(static, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        static = cv2.erode(static, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        static = cv2.dilate(static, kernel)
        static -= np.min(static)
        static = static / np.max(static) if np.max(static) != 0 else static / (np.max(static) + 0.00001)
        statics.append(torch.from_numpy(static).unsqueeze(0))
        return re
    
    return [tensor_rankpooling(arr[i:i + _w]) for i in range(l)], statics
def DynamicImage(frames, dynamic_only): # frames: [[3, 224, 224, 1], ]
    _w = args.w
    arrrp, statics = tensor_arr_rp(frames)
    arrrp = torch.cat(arrrp, dim=0) # torch.Size([64, 224, 224])
    t, h, w = arrrp.shape
    print('shapes: ', t, h, w)
    mask = torch.zeros(_w - 1, h, w)
    garrs = torch.cat((mask, arrrp), dim=0)[:t, :]
    statics = torch.cat(statics)
    statics = torch.cat((mask, statics))[:t, :]
    if dynamic_only:
        return garrs
    return (garrs + statics) * statics
def image_propose(frams):
    """
    Proposes an image based on a list of continuous frames.

    Args:
        frame_list (list): List of continuous frames.
        sl (type): index of the frames that sampled from get_sl

    Returns:
        tuple: A tuple containing two elements:
            - torch.FloatTensor: Concatenated frames.
            - torch.Tensor: Static attention maps.
    """
    skgmaparr = DynamicImage(frams, dynamic_only=False)
    return torch.cat(frams, dim=3).type(torch.FloatTensor), skgmaparr.unsqueeze(0)
"""
model = YourModel()
model.load_state_dict(torch.load('path_to_your_model.pth'))
model.eval()
"""

def put_text(frame, cls, status):
    
    
    cv2.putText(frame, f'cls: {cls}',
                (0,0+40),
                cv2.FONT_ITALIC, fontScale = 1, color = (0, 50, 200), thickness =2)
    cv2.putText(frame, f'status: {status}',
                (0,0+80),
                cv2.FONT_ITALIC, fontScale = 1, color = (200, 50, 0), thickness =2)
    
print(args)
if "Jester" in args.config:
    num_classes = 27
elif "NvGesture" in args.config:
    num_classes = 25
import onnxruntime as ort
session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
input_name = session.get_inputs()[0].name
heatmap_name = session.get_inputs()[1].name  # heatmap
output_names = [output.name for output in session.get_outputs()]


device = torch.device("cuda:0")

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
        for dx in range(60):
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
            
            if dx %2 == 0:
                tmp_time = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = transform(frame)
                frame_list.append(transform_compose(frame).view(3, sample_size, sample_size, 1))
                # process_time = time.time() - tmp_time
                # tot_time += process_time
        # print("preprocess time:", tot_time)
        # recognition_start = time.time()
        cl, skgmaparr = image_propose(frame_list)
        inputs, heatmap = cl.permute(0, 3, 1, 2), skgmaparr
        
        inputs, heatmap = map(lambda x: x.cuda(0, non_blocking=True), [inputs, heatmap])
        # input: torch.Size([1, 3, 64, 224, 224])
        # heatmap: torch.Size([1, 1, 64, 224, 224]
        inputs, heatmap = inputs.unsqueeze(0), heatmap.unsqueeze(0)
        # model(inputs, heatmap)
        inputs_np = inputs.cpu().numpy()
        heatmap_np = heatmap.cpu().numpy()
        # (logits, xs, xm, xl), distillation_loss, feature = session.run(
        #                                                         [self.session.get_]
        #                                                               )
        result = session.run(output_names, {input_name: inputs_np, heatmap_name: heatmap_np})
        logits = result[0]
        best_idx = np.argmax(logits)
        print(logits, best_idx)
        if best_idx == 12:
            gesture = "nothing"
        else:
            gesture = g_dict[best_idx]
        # recognition_end = time.time()
        # print("recognition time:", recognition_end - recognition_start)
        # print("max value:", torch.max(logits).item())
        # print(gesture,torch.argmax(logits).item())
        
        cls_ = gesture

        

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()

