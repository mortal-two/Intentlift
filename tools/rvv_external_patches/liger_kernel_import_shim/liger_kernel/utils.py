"""Report the actual non-NPU target without importing PyTorch."""


def is_npu_available() -> bool:
    return False
