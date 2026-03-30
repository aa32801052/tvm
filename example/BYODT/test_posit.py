import numpy as np
import tvm
import re
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.relax.frontend.torch import from_exported_program
from tvm import IRModule, relax
from tvm.contrib.download import download_testdata
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18
from tvm.relax.frontend.change_datatype import ChangeDatatype
from sch_handed import optimize_ir_module
from PIL import Image
import time
from register import ensure_posit_registered_for_dtype
from tvm.relax.transform import ToMixedPrecision

TARGET = "llvm"

def get_cat_image(): # Download and preprocess a cat image for testing
    url = "https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png"
    dst = "cat.png"
    real_dst = download_testdata(url, dst, module="data")
    img = Image.open(real_dst).resize((224, 224))
    
    # Preprocess the image using PyTorch's ResNet-18 weights
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    img_tensor = preprocess(img).unsqueeze(0) #(3, 224, 224) -> (1, 3, 224, 224)
    
    # Convert to numpy array with the correct data type
    img_array = img_tensor.numpy()
    return np.asarray(img_array, dtype="float32")

def export_resnet18():
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    with torch.no_grad():
        exported_program = export(model, example_args)
        relax_module = from_exported_program(exported_program, keep_params_as_input=True)
    relax_module, params = relax.frontend.detach_params(relax_module)
    relax_module = relax.transform.DecomposeOpsForInference()(relax_module)
    return relax_module, params

def _parse_posit_dtype(dtype: str):
    match = re.search(r"custom\[posites(\d+)\](\d+)", dtype)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def convert_ndarray_in_relax(dst_dtype, array, dev=tvm.cpu(), target=TARGET):
    """Convert numpy.ndarray or tvm.nd.NDArray to target dtype using Relax or posit converter"""

    if isinstance(array, list):
        return [convert_ndarray_in_relax(dst_dtype, v, dev, target) for v in array]
    if isinstance(array, dict):
        return {k: convert_ndarray_in_relax(dst_dtype, v, dev, target) for k, v in array.items()}
    
    if hasattr(array, "dtype") and hasattr(array, "shape") and not isinstance(array, np.ndarray):
        src_array = array
        src_dtype = str(src_array.dtype)
        array_np = None
    elif isinstance(array, np.ndarray):
        array_np = array
        src_array = None
        src_dtype = None
    else:
        array_np = array.numpy() if hasattr(array, 'numpy') else np.array(array)
        src_array = None
        src_dtype = None
    
    # Handle posit conversion
    posit_spec = _parse_posit_dtype(dst_dtype)
    if posit_spec is not None:
        ensure_posit_registered_for_dtype(dst_dtype)
    
    # Standard type conversion
    if src_array is None and hasattr(array_np, 'dtype'):
        if array_np.dtype == object or array_np.dtype.kind not in ['f', 'i', 'u']:
            try:
                array_np = array_np.astype(np.float32)
            except (ValueError, TypeError):
                raise ValueError(f"Cannot convert array with dtype {array_np.dtype} to numeric type")

    if src_array is None:
        src_array = tvm.runtime.tensor(array_np, dev)
        src_dtype = str(src_array.dtype)
    
    shape = src_array.shape
    mod = IRModule()
    x = relax.Var("x", relax.TensorStructInfo(shape=shape, dtype=src_dtype))
    body = relax.op.astype(x, dst_dtype)
    ret_sinfo = relax.TensorStructInfo(shape=shape, dtype=dst_dtype)
    func = relax.Function(params=[x], body=body, ret_struct_info=ret_sinfo)
    mod["main"] = func
    
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        rt_mod = relax.build(mod, target=target)
    vm = relax.VirtualMachine(rt_mod, dev)
    return vm["main"](src_array)

def ChangeDatatypeInRelaxAndParams(mod: tvm.IRModule, params: dict, src_dtype: str , dst_dtype: str):
    """
    Convert the data type of all tensors in the Relax module to the specified dtype.
    """
    # new_main = ChangeDatatype(src_dtype, dst_dtype)(mod)
    # new_mod = IRModule({"main": new_main["main"]})

    dtype_mutator = ChangeDatatype(src_dtype, dst_dtype, mod)
    new_main = dtype_mutator.visit_expr(mod["main"])
    new_mod = IRModule({"main": new_main})
    """
    Convert the data type of all input and params to the specified dtype.
    """
    params = {k: convert_ndarray_in_relax(dst_dtype, v) for k, v in params.items()}
    return new_mod, params

def run_inference(mod: tvm.IRModule, params: dict, input_data, target="llvm"):
    # Convert input data to the specified dtype
    dev = tvm.device(target, 0)
    # input_data = convert_ndarray_in_relax("float16", input_data, dev=dev)
    # Run inference on the Relax module with the given parameters and input data
    print("="*60)
    print("Start Building IRModule...")
    start_time = time.time()
    rt_mod = relax.build(mod, target=target)
    end_time = time.time()
    print("Build Time:", end_time - start_time, "seconds")
    print("-"*60)

    # Run the model
    print("Start Running Inference...")
    start_time = time.time()
    vm = relax.VirtualMachine(rt_mod, dev)
    input_data = tvm.runtime.tensor(input_data, dev)
    output = vm["main"](input_data, *params["main"])
    end_time = time.time()
    print("Inference Time:", end_time - start_time, "seconds")
    print("="*60)
    return output

def benchmark_inference_float32():
    target = "llvm"
    mod, params = export_resnet18()
    input_data = get_cat_image() 
    dev = tvm.device(target, 0)
    rt_mod = relax.build(mod, target=target)
    vm = relax.VirtualMachine(rt_mod, dev)
    input_data = tvm.runtime.tensor(input_data, dev)
    # print(params)
    output = vm["main"](input_data, *params["main"])
    return output

def main():
    src_dtype = "float32"
    dst_dtype = "custom[posites0]3"
    ensure_posit_registered_for_dtype(dst_dtype)
    input_data = get_cat_image()
    input_data = convert_ndarray_in_relax(dst_dtype=dst_dtype, array= get_cat_image())
    mod, params = export_resnet18()
    mod, params = ChangeDatatypeInRelaxAndParams(mod, params, src_dtype, dst_dtype)
    mod = tvm.relax.pipeline.get_pipeline()(mod)
    mod = optimize_ir_module(mod)
    # print(mod)
    output = run_inference(mod, params, input_data , target="llvm")
    output_f32 = convert_ndarray_in_relax(dst_dtype="float32", array=output[0])
    # print(output_f32)
    float32_out = benchmark_inference_float32()[0]
    np.testing.assert_allclose(
        float32_out.numpy(), output_f32.numpy(), rtol=1e-8, atol=1e-8
    )
if __name__ == "__main__":
    main()
