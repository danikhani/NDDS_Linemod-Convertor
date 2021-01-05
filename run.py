from prepr_linemode import make_linemode_dataset

import argparse

def parse_input():
    parser = argparse.ArgumentParser(description='Linemode dataset generator')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_input()
    make_linemode_dataset('controllercapture_try1', 'generated_linemode2', 16)

