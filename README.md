## On Synthetic Data for Back Translation

This repo is the code for NAACL2022 paper [On Synthetic Data for Back Translation](https://aclanthology.org/2022.naacl-main.32/)

## Requirements

We use python3.7, pytorch>=1.10.0, and cuda>=10.2, and [fairseq 10.2](https://github.com/pytorch/fairseq/archive/refs/tags/v0.10.2.zip)


To install fairseq, use below command
```
   cd $PROJECT_PATH
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   pip install --editable ./
   cd ..
```
Don't forget to install other packages required for back translation,
```
pip install subword_nmt
pip install -U sacremoses==0.0.41
```

Git sacremoses and subword-nmt by,
```
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
```

Since we need the script in sacremoses and subword-nmt for data preprocessing.

## A Walk Through
In this section, we have a walk through about the whole project, including back translation baselines and our proposed Gamma Sampling method

### Step1: Data Preprocessing ###

#### Parallel Data ###

In the paper, we use the wmt14 en-de and en-ru langauge pairs, the language pairs are obtained from wmt14 website, here
we have a shell script for downloading and preprocess the data. Use DE-EN as an example.

the comman used for generate the parallel databin is in the shell script

```prepare_wmt14/start.sh```

You need to indicate some variables such as ```$OUTPUT```,```$BPE_TOKENS```, etc., and this shell script could be used
for another language pairs.

For downloading, you probably wish to install mwget tools for multithread downloading.

```
wget http://jaist.dl.sourceforge.net/project/kmphpfm/mwget/0.1/mwget_0.1.0.orig.tar.bz2
tar -xjvf mwget_0.1.0.orig.tar.bz2
cd mwget_0.1.0.orig
./configure
sudo make
sudo make install
cd ..
```

#### Monolingual Data ###

The command used for generate the monolingual databin is in the shell script

```prepare_en_mono/start.sh```

Also need to indicate several path such as the  

```$OUTPUT```, ```$BPECODE ```,```$LANG``` etc.

This script will generate a path ```wmt14_en_de_full``` which contains a parallel_databin

### Step2: Baseline Training

we need to firstly train a baseline model. Take DE-EN as an example,

#### Train de-en transformer-big model as baseline ###

```
fairseq-train --fp16 \
    $WMT14_DATABIN \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --max-update 400000 \
    --save-dir $BASELINE_DE_EN_CHECKPOINT
```

where the ```$WMT14_DATABIN``` is the preprocessed databin file of wmt14 de-en bitext,
and ```$BASELINE_DE_EN_CHECKPOINT``` is the output path to save de-en checkpoint files.

#### Train en-de transformer-big model as back-translation model ###

```
fairseq-train --fp16 \
    $WMT14_DATABIN \
    --source-lang en --target-lang de \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --max-update 400000 \
    --save-dir $BASELINE_EN_DE_CHECKPOINT
```

where the ```$BASELINE_EN_DE_CHECKPOINT``` is the path to save the en-de checkpoint file, which we use to translate the
target side monolingual languages to source side. We could further evaluate the checkpoint using fairseq default
sacrebleu script

```
EVAL_DIR=fairseq/examples/backtranslation
bash $EVAL_DIR/sacrebleu.sh \
    wmt14/full \
    de-en \
    $WMT14_DATABIN \
    $WMT14_DATABIN/code \
    $DE_EN_BT_BEAM_CHECKPOINT/checkpoint_400000.pt
```

where, the ```$WMT14_DATABIN/code``` is the bpe code learned in bitext preprocess phrase.

### Step3: Back Transaltion

we use the back translation model to translate the target side languages back into source side

#### Back Translation Synthetic Corpus ###

here we can use beam or sampling method, for beam generation, using command

```
fairseq-generate \
    $MONO_DATABIN \
    --source-lang en --target-lang de \
    --path $BASELINE_EN_DE_CHECKPOINT/checkpoint_*_400000.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 25000 \
    --beam 5 \
    > beam5.out
```

for sampling, use,

```
fairseq-generate \
    $MONO_DATABIN \
    --source-lang en --target-lang de \
    --path $BASELINE_EN_DE_CHECKPOINT/checkpoint_*_400000.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 100000 \
    --beam 1 --sampling \
    > sampling.out
```

#### Extract Back Translation Data ####

Then we can extract them out using default script in fairseq. It is at
fairseq/examples/backtranslation/extract_bt_data.py

```
python3 extract_bt_data.py \
    --minlen 1 --maxlen 256 \
    --output $PATH_TO_BEAM/extracted_bt_data --srclang de --tgtlang en \
    $PATH_TO_BEAM/beam5.out
```

and we preprocess this backtranslation data and combine the back translation data with bitext parallel databin

#### Preprocess Back Translation Data ####

```
EN_DICT=$WMT14_PATH/dict.en.txt
DE_DICT=$WMT14_PATH/dict.de.txt
fairseq-preprocess \
    --source-lang de --target-lang en \
    --joined-dictionary \
    --srcdict $DE_DICT \
    --trainpref $PATH_TO_BEAM/extracted_bt_data \
    --destdir $PATH_TO_BEAM/synthetic_databin \
    --workers 128

PARA_DATA=$(readlink -f $WMT14_PATH/databin)
BT_DATA=$(readlink -f $PATH_TO_BEAM/synthetic_databin)
COMB_DATABIN=$PATH_TO_BEAM/parallel_plus_synthetic_databin
mkdir -p $COMB_DATABIN
for LANG in en de; do \
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATABIN}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATABIN}/train.de-en.$LANG.$EXT; \
        ln -s ${BT_DATA}/train.de-en.$LANG.$EXT ${COMB_DATABIN}/train1.de-en.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATABIN}/valid.de-en.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATABIN}/test.de-en.$LANG.$EXT; \
    done; \
done
```

#### Train Back Translation Forward Model ####

Finally we train and evaluate the back translation model.

```
mkdir $DE_EN_BT_BEAM_CHECKPOINT
fairseq-train \
    $COMB_DATABIN \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --max-update 1000000 \
    --save-dir $DE_EN_BT_BEAM_CHECKPOINT \
    --validate-interval-updates 10000 \
    --save-interval-updates 10000 \
    --lr 7e-4 --upsample-primary 1 --save-interval 99999999

WMT14_DATABIN=$WMT14_PATH/databin
EVAL_DIR=fairseq/examples/backtranslation
bash $EVAL_DIR/sacrebleu.sh \
    wmt14/full \
    de-en \
    $WMT14_DATABIN \
    $WMT14_DATABIN/code \
    $DE_EN_BT_BEAM_CHECKPOINT/checkpoint_best.pt
```


### Step4: Gamma

#### Train Monolingual GPT Model ###

Firstly to preprocess the single side languages of the parallel data, here the ```$DATA_DIR``` is the place where you
put the cleaned corpus from preprocess. It needs train.de valid.de and test.de three files for trainset devset and
testset, and ```$MONO_DATABIN``` is the output dir for generated databin.

```
fairseq-preprocess \
    --only-source \
    --trainpref $DATA_DIR/tmp/train.de \
    --validpref $DATA_DIR/tmp/valid.de \
    --testpref $DATA_DIR/tmp/test.de \
    --destdir $MONO_DATABIN \
    --workers 64 \
    --srcdict $DATA_DIR/databin/dict.de.txt
```

Next we use the preprocessed de-side text to train a monolingual gpt on de language.

```
DE_MONO_CHECKPOINT=checkpoints/transformer_big/de_mono_gpt
fairseq-train \
    $WMT14_DE_DATABIN \
    --task language_modeling \
    --fp16\
    --save-dir $DE_MONO_CHECKPOINT \
    --arch transformer_lm_gpt --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 4096 --update-freq 1 \
    --max-update 1000000 \
    --validate-interval-updates 10000 \
    --save-interval-updates 10000
```

Where the ```$WMT14_DE_DATABIN``` is the de side databin or, use the shell script```wmt14_de_en_de_gpt.sh```, need to
indicate the bitext data path, output databin, etc.,

#### Candidates Generation ###

We prepare a shell script ```prepare_candidates/start.sh``` for sampling 50 candidates and score each candidates with monolingual model.

You need to specify several variables such as ```$srclang```, ```$tgtlang```, etc.

If you want to do it manually, for generate candidates, use

```
fairseq-generate \
    $MONO_DATABIN \
    --source-lang en --target-lang de \
    --path $BASELINE_EN_DE_CHECKPOINT/checkpoint_*_400000.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 1000 \
    --beam 50 --sampling --nbest 50 \
    > candidates.out
```

```$MONO_DATABIN```: the monolingual corpus fairseq databin

```$BASELINE_EN_DE_CHECKPOINT```: back translation mdoel, and we sample 50 candidates

We could also split the file in shard and generate candidates within each shard. Firstly we split
the original monolingual bpe file into 30 shard, with 15,000 line per shard.
```
split --lines 150000 --numeric-suffixes \
        --additional-suffix .${tgtlang} \
        $MONO_DATA_PATH/bpe.monolingual.dedup.4500000.${tgtlang} \
        $MONO_DATA_PATH/bpe.monolingual.dedup.
```
And preprocess each shard. 
```
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
```
where the ```$CANDIDATES_PATH``` is shard databin each shard.

Then we generate 50 candidates within each shard.
```
for SHARD in $(seq -f "%02g" 0 29); do \
    fairseq-generate --fp16 \
        $CANDIDATES_PATH/shard${SHARD}/databin \
        --path $BT_MODEL_CHECKPOINT/checkpoint_*_400000.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 1024 \
        --sampling --beam 50 --nbest 50 --remove-bpe \
    > $CANDIDATES_PATH/shard${SHARD}/sampling50.shard${SHARD}.out; \
done
```



Then, we extract them out, and remove bpe code

```
for SHARD in $(seq -f "%02g" 0 29); do \
    python3 extract_bt_candidates.py \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD} \
        --srclang de --tgtlang en  \
        < $CANDIDATES_PATH/shard${SHARD}/sampling50.shard${SHARD}.out
    
    python remove_bpe.py \
        --input $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.$srclang \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.bpe_removed.$srclang
done
```

#### Scoring Importance ###

And scoring the importance with GPT,

```
for SHARD in $(seq -f "%02g" 0 29); do \
    python scoring_mono_lprob.py \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang} \
        --lm_path $MONO_MODEL_CHECKPOINT --cp_file checkpoint_best.pt \
        --databin $MONO_MODEL_DATABIN --bpe_file $WMT14_PATH/code --batch_size 50 \
        < $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}
    echo "Scoring the importance"
    python scoring_importance.py \
        --mono_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang} \
        --bt_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang} \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}
done
```

then we inline normalize the back translation score and length normalize & inline normalize importance score.

```
for SHARD in $(seq -f "%02g" 0 29); do \
echo "Normalizing the importance score"
    python length_normalize.py \
        --score_files $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.log_im_score \
        --length_files $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.seq_len \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}
    python inline_normalize.py \
        --input $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.log_im_score_len_normd \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}
    echo "inline normalize the bt_score"
    python inline_normalize.py \
        --input $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}
done
```

### Step5: Gamma Corpus ##

there are two types of gamma corpus, 

gamma selection: select the candidates with highest gamma score

gamma sampling: sample one candidates based on gamma score distribution.

#### Gamma Selection ###

For gamma selection corpus, we use

```
# construct the gamma selected corpus
GAMMA=0.2
OUTPUT_PATH=$CANDIDATES_PATH/gamma_selection${GAMMA}
mkdir -p $OUTPUT_PATH
for SHARD in $(seq -f "%02g" 28 29); do \
    python gamma_selection.py --gamma $GAMMA \
        --candidate_file $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang} \
        --target_file $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${tgtlang} \
        --candidate_imp $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.log_im_score_len_normd_inline_normd \
        --candidate_score $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score_inline_normd \
        --bt_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_lprob \
        --mono_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.mono_lprob \
        --ori_bt_score $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score \
        --ori_imp_score $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.log_im_score \
        --srclang $srclang --tgtlang $tgtlang \
        --output $OUTPUT_PATH/extracted_candidate.shard${SHARD}
done
```
Where ```$GAMMA``` is the gamma ratio which trade-off the importance and quality

#### Gamma Sampling ###

For gamma sampling corpus, we use

```
GAMMA=0.2
OUTPUT_PATH=$CANDIDATES_PATH/gamma_sampling${GAMMA}
mkdir -p $OUTPUT_PATH
for SHARD in $(seq -f "%02g" 28 29); do \
    echo "extracte shard ${SHARD}"
    python gamma_sampling.py --gamma $GAMMA \
        --candidate_file $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang} \
        --target_file $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${tgtlang} \
        --candidate_imp $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.log_im_score_len_normd_inline_normd \
        --candidate_score $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score_inline_normd \
        --bt_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_lprob \
        --mono_lprob $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.mono_lprob \
        --ori_bt_score $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.bt_score \
        --ori_imp_score $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD}.${srclang}.log_im_score \
        --srclang $srclang --tgtlang $tgtlang \
        --output $OUTPUT_PATH/extracted_candidate.shard${SHARD}
done
```

and combine the output together, and apply bpe code

```
for SHARD in $(seq -f "%02g" 0 29); do \
    echo "cat shard ${SHARD}"
    cat $OUTPUT_PATH/extracted_candidate.shard${SHARD}.de >> $OUTPUT_PATH/extracted_candidate.de
    cat $OUTPUT_PATH/extracted_candidate.shard${SHARD}.en >> $OUTPUT_PATH/extracted_candidate.en
done
python3 $PATH_TO_SUBWORDNMT/apply_bpe.py -c $WMT14_DATABIN/code \
      < $OUTPUT_PATH/extracted_candidate.de \
      > $OUTPUT_PATH/extracted_candidate.bpe_removed.de
python3 $PATH_TO_SUBWORDNMT/apply_bpe.py -c $WMT14_DATABIN/code \
      < $OUTPUT_PATH/extracted_candidate.en \
      > $OUTPUT_PATH/extracted_candidate.bpe_removed.en
```

Then we preprocess the gamma corpus, using gamma sampling as an example.

```
EN_DICT=$WMT14_PATH/databin/dict.en.txt
DE_DICT=$WMT14_PATH/databin/dict.de.txt

fairseq-preprocess \
    --source-lang $srclang --target-lang $tgtlang \
    --joined-dictionary \
    --srcdict $DE_DICT \
    --trainpref $OUTPUT_PATH/extracted_candidate.bpe_removed \
    --destdir $OUTPUT_PATH/synthetic_databin \
    --workers 128

PARA_DATA=$(readlink -f $WMT14_PATH/databin)
BT_DATA=$(readlink -f $OUTPUT_PATH/synthetic_databin)
COMB_DATABIN=$OUTPUT_PATH/parallel_plus_synthetic_databin
mkdir -p $COMB_DATABIN
for LANG in en de; do \
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATABIN}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATABIN}/train.$srclang-$tgtlang.$LANG.$EXT; \
        ln -s ${BT_DATA}/train.$srclang-$tgtlang.$LANG.$EXT ${COMB_DATABIN}/train1.$srclang-$tgtlang.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATABIN}/valid.$srclang-$tgtlang.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATABIN}/test.$srclang-$tgtlang.$LANG.$EXT; \
    done; \
done
```

### Step5: Train Gamma Model ##

Finally we train the gamma back translation model using,

```
OUTPUT_PATH=$CANDIDATES_PATH/gamma_sampling${GAMMA}
COMB_DATABIN=$OUTPUT_PATH/parallel_plus_synthetic_databin
DE_EN_GAMMA_SAMPLING_CHECKPOINT=checkpoints/transformer_big/de_en_gamma_sampling_gamma${GAMMA}
mkdir $OUTPUT_CHECKPOINT

fairseq-train --fp16 \
    $COMB_DATABIN \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 4096 --max-update 1600000 \
    --save-dir $DE_EN_GAMMA_SAMPLING_CHECKPOINT \
    --validate-interval-updates 10000 \
    --save-interval-updates 100000 \
    --lr 7e-4 --upsample-primary 1 --save-interval 99999999
```

and evaluate it by,

```
bash $EVAL_DIR/sacrebleu.sh \
    wmt14/full \
    de-en \
    $WMT14_DATABIN \
    $WMT14_DATABIN/code \
    $DE_EN_GAMMA_SAMPLING_CHECKPOINT/checkpoint_best.pt
```

## Main Experiments
We conduct the experiments on WMT14 EN-DE and EN-RU datasets, the results are shown below.

| System          | EN-DE | DE-EN | EN-RU | RU-EN |
| --------------- | ----- | ----- | ----- | ----- |
| Transformer-big | 27.4  | 32.1  | 35.9  | 34.1  |
| Beam            | 29.7  | 32.7  | 39.6  | 35.9  |
| Sampling        | 30.0  | 34.1  | 37.4  | 35.6  |
| Gamma Selection | 31.0  | 34.7  | 35.7  | 36.1  |
| Gamma Sampling  | 30.9  | 35.0  | 38.9  | 36.3  |


## Citation
```bitext
@inproceedings{xu-etal-2022-synthetic,
    title = "On Synthetic Data for Back Translation",
    author = "Xu, Jiahao  and
      Ruan, Yubin  and
      Bi, Wei  and
      Huang, Guoping  and
      Shi, Shuming  and
      Chen, Lihui  and
      Liu, Lemao",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.32",
    doi = "10.18653/v1/2022.naacl-main.32",
    pages = "419--430",
    abstract = "Back translation (BT) is one of the most significant technologies in NMT research fields. Existing attempts on BT share a common characteristic: they employ either beam search or random sampling to generate synthetic data with a backward model but seldom work studies the role of synthetic data in the performance of BT. This motivates us to ask a fundamental question: what kind of synthetic data contributes to BT performance?Through both theoretical and empirical studies, we identify two key factors on synthetic data controlling the back-translation NMT performance, which are quality and importance. Furthermore, based on our findings, we propose a simple yet effective method to generate synthetic data to better trade off both factors so as to yield the better performance for BT. We run extensive experiments on WMT14 DE-EN, EN-DE, and RU-EN benchmark tasks. By employing our proposed method to generate synthetic data, our BT model significantly outperforms the standard BT baselines (i.e., beam and sampling based methods for data generation), which proves the effectiveness of our proposed methods.",
}
```

## Questions?
If you have and questions, welcome to drop an ssues.
