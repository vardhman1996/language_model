import os
import io

def main():
    test_file = 'data/utf8.txt'
    file = io.open(test_file, 'r', encoding='utf-8')
    data = file.read().strip()

    test_data = ""
    for char in data:
        test_data += 'q' + char

    test_data += 'x'

    with io.open(test_file, 'w', encoding='utf-8') as out:
        out.write(test_data)


if __name__ == "__main__":
    main()