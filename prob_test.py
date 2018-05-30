# VALID_BMP = [(0, 0x0870), (0x089F+1, 0x1C90), (0x1CBF+1, 0x2FE0), (0x2FEF+1, 0xD800), (0xDFFF+1, 65536)]  # version 10
# def is_valid_bmp(index):
#     return any(index >= a and index < b for a,b in VALID_BMP)

file = open('output/unicode10_out.txt', 'r').readlines()
s = 0
for i, d in enumerate(file):
    try:
        s += (2 ** float(d))
    except:
        if s != 0:
            print('Prob', s)
            s = 0

print('Prob', s)



