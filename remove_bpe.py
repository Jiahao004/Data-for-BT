import fileinput
import argparse
from tqdm import tqdm

def main():
    parser=argparse.ArgumentParser(
        description=(
            "this file is used to remove the bpe code of the output"
        )
    )

    parser.add_argument("--input")

    parser.add_argument("--output")

    parser.add_argument("--bpe", default="@@")

    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as output_h:
        for line in tqdm(fileinput.input(args.input, openhook=fileinput.hook_encoded("utf-8"))):
            sents = line.rstrip().split("\t")
            bpe_removd = []
            for sent in sents:
                assert isinstance(sent, str)
                bpe_removd.append(sent.replace(args.bpe+" ",""))

            print("\t".join(bpe_removd), file=output_h)

if __name__=="__main__":
    main()
