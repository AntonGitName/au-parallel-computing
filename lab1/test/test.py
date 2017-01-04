import numpy as np
from scipy.signal import convolve2d
import subprocess
import os, errno


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured


def matrix_to_str(a):
    return '\n'.join(' '.join(str(cell) for cell in row) for row in a)


def write_test(a, b, input_file):
    input_file.write('{} {}\n'.format(a.shape[0], b.shape[0]))
    input_file.write(matrix_to_str(a))
    input_file.write('\n')
    input_file.write(matrix_to_str(b))


def read_matrix(output_file):
    content = output_file.readlines()
    n = len(content)
    return np.array(list(np.float32(x) for line in content for x in line.split())).reshape((n, n))


def convolve(a, b):
    c_ext = convolve2d(a, b)
    n, m = a.shape[0], b.shape[0]
    m2 = m //2
    return c_ext[m2:m2+n,m2:m2+n]


def check(expected, actual):
    try:
        return (np.abs(expected - actual)).max() < 1e-3
    except:
        print ('Unexpected error!')
        return False


def make_test(test):
    if not hasattr(make_test, "tests"):
        make_test.tests = []

    test_name = test.__name__
    def wrapped():
        silentremove('input.txt')
        silentremove('output.txt')

        a, b = test()
        expected = convolve(a, b)
        try:
            with open('input.txt', 'w') as input_file:
                write_test(a, b, input_file)
            subprocess.call('./convolution', shell=True)
            with open('output.txt') as output_file:
                actual = read_matrix(output_file)
                if check(expected, actual):
                    print("{}: OK".format(test_name))
                else:
                    print("{}: Fail".format(test_name))
                    print("expected: {}\n actual: {}".format(matrix_to_str(expected), matrix_to_str(actual)))
        except:
            print("{}: Fail".format(test_name))
            print("Unexpected error during the execution.")

        print()

    make_test.tests.append(wrapped)

    return wrapped


def sqr_ones(n):
    return np.ones((n, n))


@make_test
def test_5x3():
    return sqr_ones(5), sqr_ones(3)


@make_test
def test_1x9():
    return sqr_ones(1), sqr_ones(9)


@make_test
def test_31x9():
    return sqr_ones(31), sqr_ones(9)


@make_test
def test_1024x3():
    return sqr_ones(1024), sqr_ones(3)


@make_test
def test_1024x9():
    return sqr_ones(1024), sqr_ones(9)


@make_test
def test_1023x9():
    return sqr_ones(1023), sqr_ones(9)


def main():
    
    print ('tip: Please, run this tests in the same folder as executable is\n')

    for test in make_test.tests:
        test()


if __name__ == "__main__":
    main()
