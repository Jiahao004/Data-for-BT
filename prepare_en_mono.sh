
sudo ln -fs /usr/bin/python3.6 /usr/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
pip install subword_nmt
pip install -U sacremoses==0.0.41
pip install sacrebleu==1.4.13
pip install fastBPE
cd /apdcephfs/share_916081/jettexu/importance_sampling/fairseq
pip install --editable ./
cd ..


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
DEDUPLICATE=fairseq/examples/backtranslation


SUBSAMPLE_SIZE=4500000
LANG=en
orig=orig

WMT14_PATH=wmt14_en_de_full
BPE_CODE=$WMT14_PATH/code
OUTDIR=$WMT14_PATH/newscrawl_2020_${LANG}_${SUBSAMPLE_SIZE}
tmp=$OUTDIR/tmp
mkdir -p $OUTDIR $tmp
gzip -c -d $orig/news.2020.$LANG.shuffled.deduped.gz \
| shuf -n 6000000 \
| perl $NORM_PUNC $LANG \
| perl $REM_NON_PRINT_CHAR \
| perl $TOKENIZER -threads 8 -a -l $LANG \
> $tmp/monolingual.tokenized.${LANG}

# clean the corpus by length
python clean_mono_corpus.py --inputs $tmp/monolingual.tokenized.${LANG} --minlen 5 --maxlen 150 \
| shuf -n $SUBSAMPLE_SIZE \
> $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG}
wc -l $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG}

# apply BPE
python $BPEROOT/apply_bpe.py -c $BPE_CODE \
        < $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} \
        > $tmp/bpe.monolingual.${SUBSAMPLE_SIZE}.${LANG}

# remove duplicated lines
python $DEDUPLICATE/deduplicate_lines.py $tmp/bpe.monolingual.${SUBSAMPLE_SIZE}.${LANG} \
    > $OUTDIR/bpe.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG}


# preprocess the newscrawl 2020 en monolingual data
EN_DICT=$WMT14_PATH/dict.en.txt
DE_DICT=$WMT14_PATH/dict.de.txt
MONO_DATABIN=$OUTDIR/mono_databin
fairseq-preprocess \
        --only-source \
        --source-lang $LANG --target-lang de \
        --joined-dictionary \
        --srcdict $DE_DICT \
        --testpref $OUTDIR/bpe.monolingual.dedup.${SUBSAMPLE_SIZE} \
        --destdir $MONO_DATABIN \
        --workers 128
cp $EN_DICT $MONO_DATABIN
cp $DE_DICT $MONO_DATABIN
echo "done"

