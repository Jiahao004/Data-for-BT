from tqdm import tqdm
import fileinput
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate the importance based on the input monolingual loglikelihood and backtranslation loglikelihood"
        )
    )
    parser.add_argument("--mono_lprob", required=True,
                        help="the monolingual model's given loglikelihood prefix")

    parser.add_argument("--bt_lprob", required=True,
                        help="the back-translation model loglikelihood prefix")

    parser.add_argument("--output", required=True, help="the output file prefix, generate the .log_im_score file")\

    parser.add_argument("--silent_mode", default=False)

    args = parser.parse_args()

    with open(args.output + ".log_im_score", 'w', encoding="utf-8") as output_h:
        for mono_line, bt_line in tqdm(
                zip(fileinput.input(args.mono_lprob + ".mono_lprob", openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.bt_lprob + ".bt_lprob", openhook=fileinput.hook_encoded("utf-8"))),
                disable=args.silent_mode):
            mono_lprob = np.fromstring(mono_line, dtype=np.float, sep="\t")
            bt_lprob = np.fromstring(bt_line, dtype=np.float, sep="\t")
            log_im_score = mono_lprob - bt_lprob
            print("\t".join([str(x) for x in log_im_score.tolist()]), file=output_h)


if __name__ == "__main__":
    main()
