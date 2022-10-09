import argparse
import fileinput
from tqdm import tqdm
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=(
            "this file is used to normalize the score by sentence length."
        )
    )

    parser.add_argument("--score_files", nargs="*", help="the score files")

    parser.add_argument("--length_files", nargs="*", help="the length files")

    parser.add_argument("--output", required=True, help="the output length-normalized score file prefix")

    args = parser.parse_args()

    input_file_type = args.score_files[0].split(".")[-1]
    output_type = input_file_type + "_len_normd"

    with open(args.output + "." + output_type, "w", encoding="utf-8") as output_h:
        for scores, lengths in tqdm(zip(fileinput.input(args.score_files, openhook=fileinput.hook_encoded("utf-8")),
                                        fileinput.input(args.length_files, openhook=fileinput.hook_encoded("utf-8")))):
            scores = np.fromstring(scores, dtype=np.float, sep="\t")
            lengths = np.fromstring(lengths, dtype=np.float, sep="\t")
            normalized_scores = scores / lengths
            print("\t".join([str(x) for x in normalized_scores.tolist()]), file=output_h)


if __name__ == "__main__":
    main()
