def squash(vec):
    norm = vec.norm()
    norm_squared = norm ** 2
    coeff = norm_squared / (1 + norm_squared)
    return (coeff / norm) * vec

def accuracy(output, target):
    pred = output.norm(dim=0).max(0)[1].data[0]
    target = target.data[0]
    return int(pred == target)
