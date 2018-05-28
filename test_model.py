import os
import io

def main():
    test_file = 'tests/test4.txt'
    file = io.open(test_file, 'r', encoding='utf-8')
    data = file.read().strip()

    test_data = ""
    for char in data:
        test_data += 'o' + char

    print(u"" + test_data)


if __name__ == "__main__":
    main()