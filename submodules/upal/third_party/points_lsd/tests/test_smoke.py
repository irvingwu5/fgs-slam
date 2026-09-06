# 文件作用：提供points_lsd 冒烟与边界测试相关实现；隶属于points_lsd 冒烟与边界测试模块。
"""Dependency-light smoke test executed against every built wheel."""

import numpy as np
import pytest

import points_lsd


# 函数作用：执行 _square_image 对应的计算或状态操作。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：
#   - size：尺寸或数量上限。
#   - x0：x0 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - x1：x1 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - y0：y0 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - y1：y1 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def _square_image(size=200, x0=20, x1=140, y0=50, y1=150):
    image = np.zeros((size, size), np.uint8)
    image[y0:y1, x0:x1] = 255
    return image, np.array([[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]])


# 函数作用：计算符合 LSD 约定的梯度幅值和方向。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：
#   - gray：gray 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
# 输出：返回函数计算、读取或组装得到的张量、数组、标量或组合结果。
def _lsd_gradients(gray):
    """2x2 finite-difference gradient maps in the layout lsd_from_points expects.

    Same formula as ``upal.postprocess._lsd_gradients`` (original-LSD convention: value stored
    at the top-left pixel of the 2x2 stencil, angle = atan2(gx, -gy), NOTDEF = -1024).
    """
    not_defined = -1024.0
    norm = np.full(gray.shape, not_defined, np.float64)
    angle = np.full(gray.shape, not_defined, np.float64)
    a, b, c, d = gray[:-1, :-1], gray[:-1, 1:], gray[1:, :-1], gray[1:, 1:]
    gx, gy = b + d - a - c, c + d - a - b
    norm[:-1, :-1] = 0.5 * np.hypot(gx, gy)
    angle[:-1, :-1] = np.arctan2(gx, -gy)
    angle[norm <= 5.2262518595055063] = not_defined
    return norm, angle


# 函数作用：执行 _endpoints_match 对应的计算或状态操作。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：
#   - segments：segments 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - expected：expected 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
#   - tol：tol 所表示的计算输入；具体类型、形状和约束由签名与调用上下文确定。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def _endpoints_match(segments, expected, tol=3.5):
    """Every expected segment must be matched (in either orientation) by a detection.

    Detected endpoints sit ~sqrt(6) px from the ideal corners (the gradient lives one pixel
    before the intensity step and this fork omits LSD's +0.5 output shift), hence the tolerance.
    """
    assert len(segments) > 0, "no segments detected"
    for line in expected:
        forward = np.linalg.norm(segments[:, :4] - line, axis=1)
        backward = np.linalg.norm(segments[:, :4] - line[[2, 3, 0, 1]], axis=1)
        assert min(forward.min(), backward.min()) < tol, f"missing segment {line}"


# 函数作用：验证 version 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_version():
    assert isinstance(points_lsd.__version__, str) and points_lsd.__version__


# 函数作用：验证 empty 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_empty():
    assert points_lsd.lsd(np.zeros((100, 100), np.uint8)).shape == (0, 5)


# 函数作用：验证 lsd square 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_square():
    image, expected = _square_image()
    segments = points_lsd.lsd(image)
    assert segments.shape == (4, 5)
    _endpoints_match(segments, expected)


# 函数作用：验证 lsd from points square 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_from_points_square():
    x0, x1, y0, y1 = 20, 140, 50, 150
    image, expected = _square_image(x0=x0, x1=x1, y0=y0, y1=y1)
    gray = image.astype(np.float64)
    # Seeds at the middle of each edge (x, y). LSD's 2x2 forward difference places the
    # gradient on the last pixel before an intensity step, and region growing only starts
    # from seeds whose own gradient angle is defined, hence the -1 offsets.
    seeds = np.array([[80, y0 - 1], [x1 - 1, 100], [80, y1 - 1], [x0 - 1, 100]], dtype=np.int32)
    segments = points_lsd.lsd_from_points(gray, seeds, 1.0, 0.6, 0.0, *_lsd_gradients(gray))
    assert segments.ndim == 2 and segments.shape[1] >= 5
    _endpoints_match(segments, expected)


# 函数作用：验证 lsd from points requires gradients 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_from_points_requires_gradients():
    image, _ = _square_image()
    seeds = np.array([[80, 50]], dtype=np.int32)
    with pytest.raises(ValueError):
        points_lsd.lsd_from_points(image.astype(np.float64), seeds, 1.0, 0.6, 0.0)


# 函数作用：验证 lsd from points rejects out of bounds seed 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_from_points_rejects_out_of_bounds_seed():
    image, _ = _square_image()
    gray = image.astype(np.float64)
    with pytest.raises(ValueError):
        points_lsd.lsd_from_points(gray, np.array([[500, 10]], dtype=np.int32), 1.0, 0.6, 0.0, *_lsd_gradients(gray))


# 函数作用：验证 lsd from points no seeds 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_from_points_no_seeds():
    image, _ = _square_image()
    gray = image.astype(np.float64)
    segments = points_lsd.lsd_from_points(gray, np.zeros((0, 2), np.int32), 1.0, 0.6, 0.0, *_lsd_gradients(gray))
    assert len(segments) == 0


# 函数作用：验证 lsd from points rejects grad nfa 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_from_points_rejects_grad_nfa():
    image, _ = _square_image()
    gray = image.astype(np.float64)
    with pytest.raises(ValueError):
        points_lsd.lsd_from_points(gray, np.array([[80, 49]], np.int32), 1.0, 0.6, 0.0, *_lsd_gradients(gray), True)


# 函数作用：验证 lsd from points accepts non contiguous inputs 场景的预期行为。
# 所属模块：points_lsd 冒烟与边界测试。
# 输入：无。
# 输出：无显式返回值；结果通过对象状态、输出文件、绘图或断言产生。
def test_lsd_from_points_accepts_non_contiguous_inputs():
    """Strided views must be copied to C-contiguous buffers, not indexed raw."""
    x0, x1, y0, y1 = 20, 140, 50, 150
    image, expected = _square_image(x0=x0, x1=x1, y0=y0, y1=y1)
    gray = image.astype(np.float64)
    norm, angle = _lsd_gradients(gray)
    seeds = np.array([[80, y0 - 1], [x1 - 1, 100], [80, y1 - 1], [x0 - 1, 100]], dtype=np.int32)
    reference = points_lsd.lsd_from_points(gray, seeds, 1.0, 0.6, 0.0, norm, angle)

    padded = np.full((len(seeds), 3), 999, np.int64)  # int64 + hidden third column
    padded[:, :2] = seeds
    strided = points_lsd.lsd_from_points(
        np.asfortranarray(gray), padded[:, :2], 1.0, 0.6, 0.0, norm.T.copy().T, angle.astype(np.float32)
    )
    assert np.array_equal(reference, strided)
