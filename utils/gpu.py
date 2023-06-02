import torch
import gc


def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()


def get_cuda_total_memory():
    return torch.cuda.get_device_properties(0).total_memory


def _get_cuda_assumed_available_memory():
    return get_cuda_total_memory() - torch.cuda.memory_cached()


def get_cuda_available_memory():
    # Always allow for 1 GB overhead.
    return _get_cuda_assumed_available_memory() - get_cuda_blocked_memory()


def get_cuda_blocked_memory():
    # In GB steps
    available_memory = _get_cuda_assumed_available_memory()
    current_block = available_memory - 2 ** 30
    while True:
        try:
            block = torch.empty((current_block,),
                                dtype=torch.uint8, device="cuda")
            break
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                current_block -= 2 ** 30
                if current_block <= 0:
                    return available_memory
            else:
                raise
    block = None
    gc_cuda()
    return available_memory - current_block


def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )


def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def should_reduce_batch_size(exception):
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception)


def cuda_meminfo():
    print("Total:", torch.cuda.memory_allocated() / 2 ** 30, " GB Cached: ", torch.cuda.memory_cached() / 2 ** 30, "GB")
    print(
        "Max Total:",
        torch.cuda.max_memory_allocated() / 2 ** 30,
        " GB Max Cached: ",
        torch.cuda.max_memory_cached() / 2 ** 30,
        "GB",
    )