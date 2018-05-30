import os


version8 = []
with open('output/unicode8_out.txt', 'r') as f:
	for line in f:
		version8.append(line.strip())

	print(len(version8))
assert len(version8) == 316754 or len(version8) == 316754+1


version10 = []
with open('output/unicode10_out.txt', 'r') as f:
	for line in f:
		version10.append(line.strip())
	print(len(version10))
assert len(version10) == 316914 or len(version10) == 316914+1


# Generation for unicode_version 10
# Carefully check each generation char from your model by this function!
VALID_BMP = [(0, 0x0870), (0x089F+1, 0x1C90), (0x1CBF+1, 0x2FE0), (0x2FEF+1, 65536)]  # version 10
def is_valid_bmp(index):
    return any(index >= a and index < b for a,b in VALID_BMP)
