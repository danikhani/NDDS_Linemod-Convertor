from prepr_linemode import make_linemode_dataset

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Linemode dataset generator')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_input()
    make_linemode_dataset('datasets/try1', 'datasets/try1_linemode', 16)

