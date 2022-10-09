import fileinput
import argparse
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=(
            "this file is used to normalize the sequence-length-normalized importance score file with line normal normalization"
        )
    )

    parser.add_argument("--input", help="the input file for the candidates")

    parser.add_argument("--output", help="The output prefix")

    args = parser.parse_args()

    file_type = args.input.split(".")[-1]
    with open(args.output + "." + file_type+"_inline_normd", "w") as output_h:
        for line in tqdm(fileinput.input(args.input)):
            scores = np.fromstring(line.rstrip(), sep="\t")
            if len(scores)>1:
                normd_scores = (scores - scores.mean()) / scores.std()
            else:
                normd_scores = scores
            print("\t".join([str(x) for x in normd_scores.tolist()]), file=output_h)


if __name__=="__main__":
    main()
