from torchvision import transforms
import torch
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2
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
    sample_size = 224
    crop_rect = (16,16, 240, 240) # left, top, left + crop_size, top + crop_size

    img = np.asarray(img)
    if img.shape[-1] != 3:
        print("Image shape error!")
        exit()
    img = Image.fromarray(img) # 640 480
    img = img.resize((256,256)) # 256 256
    img = img.crop(crop_rect) # 224 224
    return np.array(img.resize((sample_size, sample_size)))

def tensor_arr_rp(arr):
    l = len(arr)
    statics = []
    _w = 4
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
    _w = 4
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


import tensorrt as trt
import ctypes
import torch
import numpy as np

# CUDA 드라이버 API를 로드합니다.
ctypes.cdll.LoadLibrary('libcudart.so')

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    print(chr(0x1F9E1), chr(0x1F9E2), chr(0x1F9E3))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine, batch_size=1):
    inputs = []
    outputs = []
    bindings = []
    stream = torch.cuda.Stream()

    for binding in engine:
        size = (trt.volume(engine.get_tensor_shape(binding)) * batch_size,)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))

        # Allocate host and device buffers
        host_mem = np.empty(size, dtype=dtype)
        device_mem = torch.empty(size, dtype=torch.float32, device=torch.device('cuda'))

        # Append the device buffer to device bindings.
        bindings.append(int(device_mem.data_ptr()))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, device_mem))
        else:
            outputs.append((host_mem, device_mem))

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    for host_mem, device_mem in inputs:
        device_mem.copy_(torch.from_numpy(host_mem).to(device_mem.device))

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)

    # Transfer predictions back from the GPU.
    for host_mem, device_mem in outputs:
        host_mem[:] = device_mem.cpu().numpy()

    # Synchronize the stream
    stream.synchronize()

def initialize(engine_file_path):
    # Load TensorRT engine
    engine = load_engine(engine_file_path)

    # Allocate buffers
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Create execution context
    context = engine.create_execution_context()

    return engine, context, inputs, outputs, bindings, stream

def preprocess_input(inputs, input_data_a, input_data_b):
    # Prepare input data
    np.copyto(inputs[0][0], input_data_a.ravel())
    np.copyto(inputs[1][0], input_data_b.ravel())

def main(engine_file_path):
    # Initialize
    engine, context, inputs, outputs, bindings, stream = initialize(engine_file_path)
    
    # Dummy input data for testing
    input_data_a = np.random.rand(1, 3, 64, 224, 224).astype(np.float32)
    input_data_b = np.random.rand(1, 1, 64, 224, 224).astype(np.float32)

    # Set input shapes if dynamic
    input_binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    context.set_input_shape(input_binding_names[0], input_data_a.shape)
    context.set_input_shape(input_binding_names[1], input_data_b.shape)
    import time
    while True:
        # Update dummy input data
        input_data_a = np.random.rand(1, 3, 64, 224, 224).astype(np.float32)
        input_data_b = np.random.rand(1, 1, 64, 224, 224).astype(np.float32)

        # Preprocess input
        a = time.time()
        preprocess_input(inputs, input_data_a, input_data_b)

        # Run inference
        do_inference(context, bindings, inputs, outputs, stream)

        # Reshape and return the output
        output_binding_name = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)][0]
        output_shape = context.get_tensor_shape(output_binding_name)
        output_data = np.reshape(outputs[0][0], output_shape)
        print("Inference time:", time.time() - a)
        # Print the output
        # i wanna get argmax of output data
        result = np.argmax(output_data)
        print("Inference output:", result)
        # Add a break condition or sleep if needed to control the loop
        # For example, to run the loop 10 times:
        # if some_condition_met:
        #     break
        # Or add a delay:
        # time.sleep(1)

if __name__ == "__main__":
    engine_file_path = "/home/minjae/ws/MotionRGBD/model.engine"
    main(engine_file_path)