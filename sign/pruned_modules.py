"""가지치기용 모듈. sign/prune_kd.py 가 만든 체크포인트를 읽을 때 이 파일이 import 되어 있어야 한다.

ultralytics 의 C2f(·C3k2)는 cv1 출력을 chunk 로 반으로 나눠 쓰는데, torch-pruning 은 이 나눔을 따라가지 못해
채널 번호가 어긋난다. 그래서 cv1 을 처음부터 두 개의 Conv(cv0, cv1)로 나눈 C2fSplit 으로 바꾼 뒤 자른다.
계산 결과는 바꾸기 전과 같다.
"""
from copy import deepcopy

import torch
import torch.nn as nn


class C2fSplit(nn.Module):
    def __init__(self, cv0, cv1, cv2, m):
        super().__init__()
        self.cv0, self.cv1, self.cv2, self.m = cv0, cv1, cv2, m

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _half(conv, idx):
    """ultralytics Conv(conv+bn+act)의 출력 채널 절반만 남긴 사본."""
    c = deepcopy(conv)
    out = c.conv.out_channels // 2
    sl = slice(0, out) if idx == 0 else slice(out, 2 * out)
    c.conv.weight = nn.Parameter(conv.conv.weight.data[sl].clone())
    if conv.conv.bias is not None:
        c.conv.bias = nn.Parameter(conv.conv.bias.data[sl].clone())
    c.conv.out_channels = out
    bn = c.bn
    bn.weight = nn.Parameter(conv.bn.weight.data[sl].clone())
    bn.bias = nn.Parameter(conv.bn.bias.data[sl].clone())
    bn.running_mean = conv.bn.running_mean.data[sl].clone()
    bn.running_var = conv.bn.running_var.data[sl].clone()
    bn.num_features = out
    return c


def split_c2f(model):
    """model 안의 C2f / C3k2 를 C2fSplit 으로 바꾼다. 바꾼 개수를 돌려준다."""
    from ultralytics.nn.modules.block import C2f

    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, C2f) and type(child).forward is C2f.forward:
                new = C2fSplit(_half(child.cv1, 0), _half(child.cv1, 1), child.cv2, child.m)
                for attr in ("f", "i", "type", "np"):   # ultralytics 가 층마다 붙여 두는 속성 (입력 위치 등)
                    if hasattr(child, attr):
                        setattr(new, attr, getattr(child, attr))
                setattr(mod, cname, new)
                n += 1
    return n
