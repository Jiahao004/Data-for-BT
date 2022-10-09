
import argparse
import fileinput
from fairseq.models.transformer import TransformerModel
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract back-translations from the stdout of fairseq-generate. "
            "If there are multihypoes for a source sentence, we cat all of them with tab. "
        )
    )
    parser.add_argument("--output", required=True, help="output prefix")

    parser.add_argument("--srclang", required=True, help="source language (extracted from H-* lines)")

    parser.add_argument("--tgtlang", required=True, help="target language (extracted from S-* lines)")

    parser.add_argument("--maxlen", type=int, help="the maximum token length for each candidate")

    parser.add_argument("--minlen", type=int, help="the minimum token length for each candidate")

    parser.add_argument("--first_only", default=False, action="store_true", help="set to extract only the first candiate")

    parser.add_argument("files", nargs="*", help="input files")
    args = parser.parse_args()

    def safe_index(toks, index, default):
        try:
            return toks[index]
        except IndexError:
            return default

    count = 0
    count_s, count_d, count_t = 0, 0, 0
    true_src = False
    with open(args.output + "." + args.srclang, "w", encoding='utf-8') as src_h, \
            open(args.output + "." + args.srclang + ".bt_score", "w", encoding="utf-8") as src_score_h, \
            open(args.output + "." + args.srclang + ".bt_lprob", "w", encoding="utf-8") as src_lprob_h, \
            open(args.output + "." + args.srclang + ".token_lprob", "w", encoding="utf-8") as src_token_lprob_h, \
            open(args.output + "." + args.srclang + ".seq_len", "w", encoding="utf-8") as src_seq_len, \
            open(args.output + "." + args.tgtlang, "w", encoding='utf-8') as tgt_h, \
            open(args.output + "." + args.srclang + ".src", "w", encoding="utf-8") as true_src_h:
        start = True
        for line in tqdm(fileinput.input(args.files, openhook=fileinput.hook_encoded('utf-8'))):
                if line.startswith("S-"):
                    count_s += 1
                    if not start:
                        line_src, line_bt_score, line_bt_lprob, line_token_lprob, line_seq_len = \
                            remove_duplicates(line_src, line_bt_score, line_bt_lprob, line_token_lprob, line_seq_len)
                        n_line_src = len(line_src)
                        assert n_line_src == len(line_bt_score)
                        assert n_line_src == len(line_bt_lprob)
                        assert n_line_src == len(line_token_lprob)
                        assert n_line_src == len(line_seq_len)
                        if args.first_only:
                            print(line_src[0], file=src_h)
                            print(line_bt_score[0], file=src_score_h)
                            print(line_bt_lprob[0], file=src_lprob_h)
                            print(line_token_lprob[0], file=src_token_lprob_h)
                            print(line_seq_len[0], file=src_seq_len)
                        else:
                            print("\t".join(line_src), file=src_h)
                            print("\t".join(line_bt_score), file=src_score_h)
                            print("\t".join(line_bt_lprob), file=src_lprob_h)
                            print("\t".join(line_token_lprob), file=src_token_lprob_h)
                            print("\t".join(line_seq_len), file=src_seq_len)
                        print(tgt, file=tgt_h)
                        count += 1
                        if true_src:
                            print(true_src, file=true_src_h)

                    start = False
                    line_src = []
                    line_bt_score = []
                    line_bt_lprob = []
                    line_token_lprob=[]
                    line_seq_len = []
                    tgt = safe_index(line.rstrip().split("\t"), 1, "")
                    len_tgt = len(tgt.split())

                elif line.startswith("H-"):
                    count_d += 1
                    if tgt is not None:
                        splited_line = line.rstrip().split("\t")
                        score = safe_index(splited_line, 1, "")
                        src = safe_index(splited_line, 2, "")

                        # filter out the candidates which are shorter than minlen or longer than maxlen
                        if args.minlen and args.maxlen:
                            if args.minlen < len(src.split()) < args.maxlen and args.minlen < len_tgt < args.maxlen:
                                line_src.append(src)
                                line_bt_score.append(score)
                                # adding 2 because each sequence has <SOS> and <EOS> tokens to indicate start-of-speach and end-of-speach
                                seq_len = 2 + len(src.split())
                                line_seq_len.append(str(seq_len))

                        else:
                            line_src.append(src)
                            line_bt_score.append(score)
                            # adding 2 because each sequence has <SOS> and <EOS> tokens to indicate start-of-speach and end-of-speach
                            seq_len = 2 + len(src.split())
                            line_seq_len.append(str(seq_len))
                elif line.startswith("P-"):
                    splited_line = line.rstrip().split("\t")
                    score = safe_index(splited_line, 1,"")
                    if args.minlen and args.maxlen:
                        if args.minlen < len(src.split()) < args.maxlen and args.minlen < len_tgt < args.maxlen:
                            line_bt_lprob.append(str(sum([eval(x) for x in score.split()])))
                            line_token_lprob.append(score)
                    else:
                        line_bt_lprob.append(str(sum([eval(x) for x in score.split()])))
                        line_token_lprob.append(score)

                elif line.startswith("T-"):
                    count_t += 1
                    true_src = safe_index(line.rstrip().split("\t"), 1, "")
                elif line.startswith("BT-"):
                    raise Exception(f"this file contains BT- lines, use extract_bt_imp_sampling_candidates.py instead")

        if len(line_src) != 0:
            print("\t".join(line_src), file=src_h)
            print("\t".join(line_bt_score), file=src_score_h)
            print("\t".join(line_bt_lprob), file=src_lprob_h)
            print("\t".join(line_token_lprob), file=src_token_lprob_h)
            print("\t".join(line_seq_len), file=src_seq_len)
            print(tgt, file=tgt_h)
            print(true_src, file=true_src_h)
            count+=1


        print(f"READIN S:{count_s}-D:{count_d}T:{count_t}")
        print(f"WRITEOUT {count} sentences")


def remove_duplicates(line_src, line_bt_score, line_bt_lprob, line_token_lprob,line_seq_len):
    # remove the duplicates
    output_src, output_bt_score, output_bt_lprob, output_token_lprob,output_seq_len = [], [], [], [],[]
    hash_map = set()
    for i in range(len(line_src)):
        hash_value = hash(line_src[i])
        if hash_value not in hash_map:
            hash_map.add(hash_value)
            output_src.append(line_src[i])
            output_bt_score.append(line_bt_score[i])
            output_bt_lprob.append(line_bt_lprob[i])
            output_token_lprob.append(line_token_lprob[i])
            output_seq_len.append(line_seq_len[i])
    return output_src, output_bt_score, output_bt_lprob, output_token_lprob,output_seq_len


if __name__ == "__main__":
    main()
