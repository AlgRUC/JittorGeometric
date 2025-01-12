import jittor as jt
def repeat_interleave(x, repeats, dim=None):
    if isinstance(repeats, int):
        if dim is None:
            x = x.reshape(-1)
            dim = 0
        if dim < 0:
            dim += x.ndim

        # 计算目标形状
        tar_shape = list(x.shape)
        tar_shape[dim] = tar_shape[dim] * repeats
        dims = []
        for i in range(len(tar_shape)):
            if dim == i:
                dims.append(f"i{i}/{repeats}")
            else:
                dims.append(f"i{i}")
        return x.reindex(tar_shape, dims)

    elif isinstance(repeats, jt.Var):
        # 检查 repeats 在指定维度上的大小是否与输入张量一致
        if dim is None:
            raise ValueError("When repeats is a jt.Var, dim must be specified.")
        if dim < 0:
            dim += x.ndim
        if repeats.shape[0] != x.shape[dim]:
            raise ValueError(f"repeats must have the same size as input along dimension {dim}.")

        result = []
        # 对指定维度进行逐个元素重复
        for i in range(x.shape[dim]):
            # 提取切片，获取第 i 个元素
            slice_obj = [slice(None)] * x.ndim
            slice_obj[dim] = slice(i, i + 1)
            sliced_x = x[tuple(slice_obj)]

            expanded_x = sliced_x
            for _ in range(repeats[i].item() - 1):  # 重复 repeats[i] - 1 次
                expanded_x = jt.concat([expanded_x, sliced_x], dim=dim)  # 沿指定维度拼接
            result.append(expanded_x)  # 将扩展后的元素加入结果列表
        result = jt.concat(result, dim=dim)

        return result
    else:
        raise ValueError("repeats should be either int or jt.Var")


