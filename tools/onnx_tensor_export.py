
import argparse
import os
import sys
from pathlib import Path
import torch
sys.path.append(os.path.join('..', os.path.abspath(os.path.join(os.getcwd()))) )
 
from lib import *
from utils import *
def export_onnx(model, device, file, opset):
    
    import onnx
    
    f = "model.onnx"
    im = torch.empty(1, 3, 64, 224, 224).to(device)
    heatmap = torch.empty(1, 1, 64, 224, 224).to(device)
    torch.onnx.export(
        model,  # --dynamic only compatible with cpu
        (im, heatmap),
        f,
        verbose=False,
        opset_version=14,
        do_constant_folding=True,
        input_names=['inputs', 'heatmap'],
        output_names=["output"],
    )
    
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    onnx.save(model_onnx, f)
    print("did i success?")



def export_engine(model, weight_path,device, workspace=4, verbose=False):
    
    import tensorrt as trt

    # export_onnx(model, im, file, 12, dynamic, half, simplify)  # opset 13
    f = "/home/minjae/ws/MotionRGBD/model.engine"  # TensorRT engine file
    '''
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH):
    네트워크 정의 생성 시 사용할 플래그를 설정합니다. EXPLICIT_BATCH는 명시적으로 배치 차원을 처리하도록 설정하는 플래그입니다. 이는 ONNX 모델에서 배치 크기를 동적으로 조절할 수 있게 해주며, 이 플래그가 설정되면 네트워크의 입력 차원에 배치 크기가 포함됩니다.
    network = builder.create_network(flag):
    flag를 사용하여 네트워크 정의를 생성합니다. 이 네트워크는 나중에 ONNX 모델을 읽어들여 TensorRT 엔진으로 변환할 기반을 제공합니다.
    parser = trt.OnnxParser(network, logger):
    ONNX 모델을 읽어들여 TensorRT 네트워크 정의로 변환하는 OnnxParser 객체를 생성합니다. 이 파서는 ONNX 파일의 구조를 분석하고, TensorRT에서 사용할 수 있는 네트워크 구조로 변환하는 역할을 합니다. 생성자에 network와 logger를 전달하여 파싱 과정에서의 네트워크 구조와 로그 처리 방식을 지정합니다.
    이 과정을 통해 ONNX 모델을 TensorRT 엔진으로 변환하는 데 필요한 준비 작업을 완료하게 됩니다. 이후 모델 파일을 읽어들이고 파싱하여 최적화된 인퍼런스 엔진을 구축할 수 있습니다.
    '''
    logger = trt.Logger(trt.Logger.INFO)
    logger.min_severity = trt.Logger.Severity.VERBOSE
    onnx = "/home/minjae/ws/MotionRGBD/model.onnx"
    builder = trt.Builder(logger) # create builder
    config = builder.create_builder_config() # create builder config
    config.max_workspace_size = workspace * 1 << 30

    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # if trt has problem, i may delete this
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    parser.parse_from_file(str(onnx))
    

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    logger.info("Network Description:")
    for inp in inputs:
        logger.info(
            f'\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}'
        )
    for out in outputs:
        logger.info(
            f'\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}'
        )

    with builder.build_engine(network, config) as engine, open(f, "wb") as t:
        t.write(engine.serialize())
    return f
import torch
import tensorrt as trt

def export_engine2(model, weight_path, device, workspace=4, verbose=False):
    # Initialize input tensors
    
    # TensorRT engine file path
    engine_file_path = "/home/minjae/ws/MotionRGBD/model.engine"
    
    # Initialize TensorRT logger
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    
    # ONNX model file path
    onnx_file_path = "/home/minjae/ws/MotionRGBD/model.onnx"
    
    # Create TensorRT builder, network and parser
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * (1 << 30)  # Set max workspace size
    
    # Create network with explicit batch flag
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    if not parser.parse_from_file(onnx_file_path):
        for error in range(parser.num_errors):
            logger.log(trt.Logger.ERROR, parser.get_error(error).desc())
        return None
    
    # Log network inputs and outputs
    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    logger.log(trt.Logger.INFO, "Network Description:")
    for inp in inputs:
        logger.log(trt.Logger.INFO, f'\tinput "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
    for out in outputs:
        logger.log(trt.Logger.INFO, f'\toutput "{out.name}" with shape {out.shape} and dtype {out.dtype}')
    
    logger.log(trt.Logger.INFO, "All layers and their output shapes:")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        logger.log(trt.Logger.INFO, f'Layer {i}: {layer.name} ({layer.type})')
        for j in range(layer.num_outputs):
            output = layer.get_output(j)
            logger.log(trt.Logger.INFO, f'    Output {j}: shape={output.shape}, dtype={output.dtype}')
    
    # Build serialized engine and save to file
    serialized_engine = builder.build_serialized_network(network, config)
    print("??")
    if serialized_engine is None:
        logger.log(trt.Logger.ERROR, "Failed to build engine.")
        return None
    
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    
    return engine_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID export")
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
    parser.add_argument(
        "--weights",
        type=Path,
        default= "osnet_x0_25_msmt17.pt",
        help="model.pt path(s)",
    )
    args = parser.parse_args()
    from config import Config

    args = Config(args)
    

    device = torch.device("cuda:0")
    if "Jester" in args.config:
        num_classes = 27
    else:
        num_classes = 25
 
    model = DSNNet(args, num_classes = num_classes, pretrained=args.weights, phase = "valid")
    model.to(device)
    model.eval()
    weight_path = "model.onnx"
    export_onnx(model, device,  args.weights, 12)  # opset 12
    # export_engine2(model, weight_path, device)