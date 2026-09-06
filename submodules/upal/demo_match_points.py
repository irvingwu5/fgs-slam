# 文件作用：提供点特征匹配演示相关实现；隶属于点特征匹配演示模块。
#!/usr/bin/env python3
"""Match UPAL point descriptors between two images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from upal import load_model
from upal.demo_utils import draw_features, load_image, side_by_side
from upal.postprocess import mutual_nearest_neighbors


# 函数作用：解析参数并运行当前演示入口。
# 所属模块：点特征匹配演示。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", type=Path, default=Path("assets/boat1.png"))
    parser.add_argument("--image1", type=Path, default=Path("assets/boat2.png"))
    parser.add_argument("--weights", type=Path, default=Path("weights/upal.tar"))
    parser.add_argument("--output", type=Path, default=Path("outputs/point_matches.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=100)
    args = parser.parse_args()

    model = load_model(args.weights, device=args.device, max_num_keypoints=args.max_keypoints)
    images, predictions = [], []
    for path in (args.image0, args.image1):
        image, tensor = load_image(path, args.max_size)
        with torch.inference_mode():
            prediction = model(tensor.to(args.device))
        images.append(image)
        predictions.append(prediction)
    matches = mutual_nearest_neighbors(predictions[0]["descriptors"][0], predictions[1]["descriptors"][0])
    similarity = (predictions[0]["descriptors"][0, matches[:, 0]] * predictions[1]["descriptors"][0, matches[:, 1]]).sum(1)
    matches = matches[similarity.argsort(descending=True)[: args.max_matches]]

    canvas, offset = side_by_side(*[draw_features(image, keypoints=prediction["keypoints"][0].cpu().numpy()) for image, prediction in zip(images, predictions)])
    rng = np.random.default_rng(7)
    for index0, index1 in matches.cpu().numpy():
        p0 = tuple(np.rint(predictions[0]["keypoints"][0, index0].cpu().numpy()).astype(int))
        p1 = np.rint(predictions[1]["keypoints"][0, index1].cpu().numpy()).astype(int)
        cv2.line(canvas, p0, (int(p1[0] + offset), int(p1[1])), tuple(int(v) for v in rng.integers(80, 256, 3)), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"mutual point matches shown: {len(matches)}")
    print(f"visualization: {args.output}")


if __name__ == "__main__":
    main()
