sudo ln -fs /usr/bin/python3.6 /usr/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
pip install subword_nmt
pip install -U sacremoses==0.0.41
pip install sacrebleu==1.4.13
pip install fastBPE tensorboardX
cd /apdcephfs/share_916081/jettexu/importance_sampling/fairseq
pip install --editable ./
cd ..


# this document contains the equal-scale monolingual corpus with wmt14
srclang=de
tgtlang=en
WMT14_PATH=wmt14_en_de_full
MONO_DATA_PATH=$WMT14_PATH/newscrawl_2020_${tgtlang}_4500000
MONO_DATABIN=$MONO_DATA_PATH/mono_databin
BT_MODEL_CHECKPOINT=checkpoints/transformer_big/${tgtlang}_${srclang}_baseline
MONO_MODEL_CHECKPOINT=checkpoints/transformer_big/de_mono_gpt

split --lines 150000 --numeric-suffixes \
        --additional-suffix .${tgtlang} \
        $MONO_DATA_PATH/bpe.monolingual.dedup.4500000.${tgtlang} \
        $MONO_DATA_PATH/bpe.monolingual.dedup.

# preprocess the monodata

CANDIDATES_PATH=$MONO_DATA_PATH/candidates
mkdir -p $CANDIDATES_PATH
for SHARD in $(seq -f "%02g" 0 29); do \
    fairseq-preprocess \
        --only-source \
        --source-lang ${tgtlang} --target-lang ${srclang} \
        --joined-dictionary \
        --srcdict $WMT14_PATH/dict.${tgtlang}.txt \
        --testpref $MONO_DATA_PATH/bpe.monolingual.dedup.${SHARD} \
        --destdir $CANDIDATES_PATH/shard${SHARD} \
        --workers 32; \
    cp $WMT14_PATH/dict.${srclang}.txt $CANDIDATES_PATH/shard${SHARD}/; \
done

# Generate the 100 candidates
echo "Eval BT model.."
WMT14_DATABIN=$WMT14_PATH/databin
EVAL_PATH=fairseq/examples/backtranslation/
bash $EVAL_PATH/sacrebleu.sh \
    wmt14/full \
    ${tgtlang}-${srclang} \
    $WMT14_DATABIN \
    $WMT14_PATH/code \
    $BT_MODEL_CHECKPOINT/checkpoint_*_400000.pt
#EN_RU "63.6/41.3/29.1/21.0 (BP = 0.998 ratio = 0.998 hyp_len = 61474 ref_len = 61603)"
#BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.13 = 27.5 58.1/33.0/21.2/14.0 (BP = 1.000 ratio = 1.035 hyp_len = 64910 ref_len = 62688)

SHARD=0
CANDIDATES_PATH=$MONO_DATA_PATH/candidates
MONO_MODEL_DATABIN=$WMT14_PATH/${srclang}_mono_databin
for SHARD in $(seq -f "%02g" 0 29); do \
    fairseq-generate --fp16 \
        $CANDIDATES_PATH/shard${SHARD} \
        --path $BT_MODEL_CHECKPOINT/checkpoint_*_400000.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 1024 \
        --sampling --beam 50 --nbest 50 \
    > $CANDIDATES_PATH/shard${SHARD}/sampling50.shard${SHARD}.out; \
    echo "Start to extract the 50 candidates out..."
    python extract_bt_candidates.py \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD} \
        --srclang $srclang --tgtlang $tgtlang  \
        --minlen 1 --maxlen 150 \
        < $CANDIDATES_PATH/shard${SHARD}/sampling50.shard${SHARD}.out
    wc -l $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score
    python remove_bpe.py \
        --input $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.$srclang \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.$srclang

    echo "Start to scoring the 50 candidates by monolingual model"
    python scoring_mono_lprob.py \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang} \
        --lm_path $MONO_MODEL_CHECKPOINT --cp_file checkpoint_best.pt \
        --databin $MONO_MODEL_DATABIN --bpe_file $WMT14_PATH/code --batch_size 50 \
        < $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang}

    echo "Scoring the importance"
    python scoring_importance.py \
        --mono_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang} \
        --bt_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang} \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang}
    echo "Normalizing the importance score"
    python length_normalize.py \
        --score_files $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang}.log_im_score \
        --length_files $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.seq_len \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang}
    python inline_normalize.py \
        --input $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang}.log_im_score_len_normd \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.${srclang}
    echo "inline normalize the bt_score"
    python inline_normalize.py \
        --input $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}
done


# evaluate the language modelling model
fairseq-eval-lm $WMT14_PATH/${srclang}_mono_databin \
    --path $MONO_MODEL_CHECKPOINT/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400



