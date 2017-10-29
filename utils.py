def squash(vec):
    norm = vec.norm()
    norm_squared = norm ** 2
    coeff = norm_squared / (1 + norm_squared)
    return (coeff / norm) * vec
