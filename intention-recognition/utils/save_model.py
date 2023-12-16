import numpy
import torch
import onnxruntime as ort
from torch.onnx import TrainingMode


def to_jit():
    model_path = '../model/datas/model/net_30.pkl'
    mod = torch.load(model_path, map_location='cpu')
    mod.eval()
    mod = torch.jit.trace(mod, torch.randint(100, size=(2, 20)))  # 转换成静态模型
    torch.jit.save(mod, 'mod.pt')  # 静态模型保存


def tt_jit():
    model_path = '../model/datas/model/net_30.pkl'
    mod = torch.load(model_path, map_location='cpu')
    mod.eval()
    jit_mod = torch.jit.load('mod.pt', map_location='cpu')
    jit_mod.eval()
    # 测试
    x = torch.randint(100, (2, 20))
    z1 = mod(x)
    z2 = jit_mod(x)
    print(torch.mean(torch.abs(z1 - z2)))


def to_onnx():
    model_path = '../model/datas/model/net_30.pkl'
    mod = torch.load(model_path, map_location='cpu')
    mod.eval()
    torch.onnx.export(
        model=mod,  # 输出的模型
        args=(torch.randint(100, size=(1, 23)),),  # 案例
        f='mod_01.onnx',  # 路径
        training=TrainingMode.EVAL,
        verbose=False,  # 是否打印日志
        input_names=['ip_name'],  # 输入的名称
        output_names=['op_name'],  # 输出的名称
        dynamic_axes=None  # 如果设置为None，模型导入后必须和案例的形状相同
    )


def to_onnx1():
    model_path = '../model/datas/model/net_30.pkl'
    mod = torch.load(model_path, map_location='cpu')
    mod.eval()
    torch.onnx.export(
        model=mod,  # 输出的模型
        args=(torch.randint(100, size=(1, 23)),),  # 案例
        f='mod_02.onnx',  # 路径
        verbose=False,  # 是否打印日志
        training=TrainingMode.EVAL,
        input_names=['ip_name'],  # 输入的名称
        output_names=['op_name'],  # 输出的名称
        dynamic_axes={
            'ip_name': {
                0: 'n',  # 表示输入的第一个维度是不确定的，可以是n维
                1: 't'
            },
            'op_name': {
                0: 'n'
            }
        }  # 如果设置为None，模型导入后必须和案例的形状相同
    )


def tt_onnx():
    model_path = '../model/datas/model/net_30.pkl'
    mod = torch.load(model_path, map_location='cpu')
    mod.eval()
    device = ort.get_device()
    provide = ort.get_available_providers()
    print(device)
    print(provide)
    onnx1 = ort.InferenceSession('mod_01.onnx',
                                 providers=['CUDAExecutionProvider',
                                            'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
                                     'CPUExecutionProvider'])
    onnx2 = ort.InferenceSession('mod_02.onnx',
                                 providers=['CUDAExecutionProvider',
                                            'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
                                     'CPUExecutionProvider'])
    output = onnx1.get_outputs()[0]
    print(output.name)
    x = torch.randint(100, (1, 23))
    x2 = torch.randint(100, (2, 23))
    mod1 = mod(x)
    mod2 = mod(x2)
    mod11 = onnx1.run(output_names=[output.name], input_feed={
        'ip_name': x.detach().numpy()
    })[0]
    try:
        mod12 = onnx1.run(output_names=[output.name], input_feed={
            'ip_name': x2.detach().numpy()
        })[0]
    except Exception as e:
        print(f'异常{e}')

    mod21 = onnx2.run([onnx2.get_outputs()[0].name], input_feed={onnx2.get_inputs()[0].name: x.detach().numpy()})
    mod22 = onnx2.run([onnx2.get_outputs()[0].name], input_feed={onnx2.get_inputs()[0].name: x2.detach().numpy()})

    print(numpy.mean(numpy.abs(mod11 - mod1.detach().numpy())))
    print(numpy.mean(numpy.abs(mod21 - mod1.detach().numpy())))
    print(numpy.mean(numpy.abs(mod22 - mod2.detach().numpy())))


if __name__ == '__main__':
    # to_jit()
    # tt_jit()
    # to_onnx()
    # to_onnx1()
    tt_onnx()
