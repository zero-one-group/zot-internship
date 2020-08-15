"What is the index of the first term in the Fibonacci sequence to contain 1000 digits?"""


def fib_sequence():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a + b


def first_n_digit_in_fib(digit):
    fib = enumerate(fib_sequence())
    x = 1
    while len(str(x)) < digit:
        i, x = next(fib)
    return i+1


print("index of sequence with first 1000 digit is ",
      first_n_digit_in_fib(1000))
