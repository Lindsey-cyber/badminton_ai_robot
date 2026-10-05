"""Check the static YOLOv8 one-class ONNX layout expected by the demo."""

from pathlib import Path


def check_detection_layout(input_shape: list, output_shape: list,
                           image_size: int) -> None:
    if input_shape != [1, 3, image_size, image_size]:
        raise ValueError(f"Expected fixed BCHW input [1, 3, {image_size}, {image_size}], got {input_shape}")
    if (len(output_shape) != 3 or output_shape[0] != 1 or
            output_shape[1] != 5 or not isinstance(output_shape[2], int) or
            output_shape[2] < 1):
        raise ValueError(f"Expected raw one-class YOLOv8 output [1, 5, N], got {output_shape}")


def check_onnx_file(path: Path, image_size: int) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("The existing detector requires one input and one raw output")
    check_detection_layout(session.get_inputs()[0].shape,
                           session.get_outputs()[0].shape, image_size)
