import fileinput
import numpy as np
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "this file is used to select the candidate based on gamma"
            "candidate=argmax{gamma*imp+(1-gamma)*bt_score}"
        )
    )

    parser.add_argument("--candidate_file", nargs="*")

    parser.add_argument("--target_file", nargs="*")

    parser.add_argument("--gamma", type=float, default=0.2)

    parser.add_argument("--candidate_imp", nargs="*")

    parser.add_argument("--candidate_score", nargs="*")

    parser.add_argument("--bt_lprob", nargs="*")

    parser.add_argument("--mono_lprob", nargs="*")

    parser.add_argument("--ori_bt_score", nargs="*")

    parser.add_argument("--ori_imp_score", nargs="*")

    parser.add_argument("--srclang")
    parser.add_argument("--tgtlang")

    parser.add_argument("--output", help="the output prefix")

    args = parser.parse_args()

    ori=getattr(args,"ori_bt_score",False)
    srclang = args.srclang
    tgtlang = args.tgtlang
    gamma = args.gamma
    with open(args.output + "." + srclang, "w", encoding="utf-8") as src_h, \
            open(args.output + "." + "gamma", "w", encoding="utf-8") as gamma_h, \
            open(args.output + "." + "index", "w", encoding="utf-8") as index_h, \
            open(args.output + "." + tgtlang, "w", encoding="utf-8") as tgt_h, \
            open(args.output + "." + srclang + ".ori_imp", "w", encoding="utf-8") as ori_imp_h, \
            open(args.output + "." + srclang + ".ori_bt_lprob", "w", encoding="utf-8") as ori_bt_lprob_h, \
            open(args.output + "." + srclang + ".ori_mono_lprob", "w", encoding="utf-8") as ori_mono_lprob_h:
        print("------------imp_score|bt_score|gamma_score|bt_lprob|mono_lprob|ori_bt_score|ori_imp-------------",
              file=gamma_h)
        for imps, scores, candidates, bt_lprobs, mono_lprobs, ori_bts, ori_imps, target in tqdm(
                zip(fileinput.input(args.candidate_imp, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.candidate_score, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.candidate_file, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.bt_lprob, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.mono_lprob, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.ori_bt_score, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.ori_imp_score, openhook=fileinput.hook_encoded("utf-8")),
                    fileinput.input(args.target_file, openhook=fileinput.hook_encoded("utf-8")))):
            imps, scores = np.fromstring(imps.rstrip(), sep="\t"), np.fromstring(scores.rstrip(), sep="\t")
            bt_lprobs, mono_lprobs = np.fromstring(bt_lprobs.rstrip(), sep="\t"), np.fromstring(mono_lprobs.rstrip(),
                                                                                                sep="\t")
            ori_bts, ori_imps = np.fromstring(ori_bts.rstrip(), sep="\t"), np.fromstring(ori_imps.rstrip(), sep="\t")

            gamma_scores = gamma * imps + (1 - gamma) * scores
            # TODO change the selection method from selecting the maximum value of gamma score to sampling by gamma score distribution
            # index = np.argmax(gamma_scores)
            gamma_prob = softmax(np.exp(gamma_scores))
            index=np.random.choice(len(gamma_scores),1,p=gamma_prob)[0]
            print(index, file=index_h)
            print(candidates.rstrip().split("\t")[index], file=src_h)
            print(target.rstrip(), file=tgt_h)
            assert mono_lprobs[index] != ori_imps[index]
            assert bt_lprobs[index] != ori_imps[index]
            print(
                f"{imps[index]}|{scores[index]}|{gamma_scores[index]}|{bt_lprobs[index]}|{mono_lprobs[index]}|{ori_bts[index]}|{ori_imps[index]}",
                file=gamma_h)
            print(bt_lprobs[index], file=ori_bt_lprob_h)
            print(mono_lprobs[index], file=ori_mono_lprob_h)
            print(ori_imps[index], file=ori_imp_h)


def softmax(x,tau=1):
    return np.exp(x/tau)/np.sum(np.exp(x/tau))

if __name__ == "__main__":
    main()
