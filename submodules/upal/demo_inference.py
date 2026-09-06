# 文件作用：提供单图推理演示相关实现；隶属于单图推理演示模块。
#!/usr/bin/env python3
"""Run UPAL on one image and render its detected points and lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

from upal import load_model
from upal.demo_utils import draw_features, load_image
from upal.postprocess import detect_lines


# 函数作用：解析参数并运行当前演示入口。
# 所属模块：单图推理演示。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("assets/boat1.png"))
    parser.add_argument("--weights", type=Path, default=Path("weights/upal.tar"))
    parser.add_argument("--output", type=Path, default=Path("outputs/inference.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    args = parser.parse_args()

    image, tensor = load_image(args.image, args.max_size)
    model = load_model(args.weights, device=args.device, max_num_keypoints=args.max_keypoints)
    model_input = tensor.to(args.device)
    with torch.inference_mode():
        prediction = model(model_input)
    lines = detect_lines(
        model_input,
        prediction["line_distance_field"][0],
        prediction["keypoints"][0],
    )

    visualization = draw_features(
        image,
        keypoints=prediction["keypoints"][0].cpu().numpy(),
        lines=lines,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), visualization)
    print(f"keypoints: {prediction['keypoints'].shape[1]}; point-seeded lines: {len(lines)}")
    print(f"visualization: {args.output}")


if __name__ == "__main__":
    main()
