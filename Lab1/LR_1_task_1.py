def or_func(x1, x2):
    w1, w2, w0 = 1, 1, -0.5
    return 1 if (w1*x1 + w2*x2 + w0) >= 0 else 0

def and_func(x1, x2):
    w1, w2, w0 = 1, 1, -1.5
    return 1 if (w1*x1 + w2*x2 + w0) >= 0 else 0

def xor_func(x1, x2):
    y1 = or_func(x1, x2)
    y2 = and_func(x1, x2)
    return 1 if (y1 - y2) >= 0.5 else 0

print("x1 | x2 | OR | AND | XOR")
print("-" * 30)
for x1 in [0, 1]:
    for x2 in [0, 1]:
        or_result = or_func(x1, x2)
        and_result = and_func(x1, x2)
        xor_result = xor_func(x1, x2)
        print(f" {x1} |  {x2} |  {or_result} |  {and_result}  |  {xor_result}")