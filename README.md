# importance-back-translation

The implementation of paper on synthetic data for back translation

# Requirements

we use python3.7, pytorch>=1.10.0, and cuda>=10.2

[comment]: <> (, or using docker from ```mirrors.tencent.com/jh_xu/g-tlinux2.2-python3.6-cuda11.0-cudnn8.1:latest```)

For packages, 
a [fairseq 10.2](https://github.com/pytorch/fairseq/archive/refs/tags/v0.10.2.zip) is needed

[comment]: <> (This also could be found in path ```/apdcephfs/share_916081/jettexu/importance_sampling/fairseq```)

To install fairseq, 
```
   cd $PROJECT_PATH
   git clone https://github.com/pytorch/fairseq
   cd fairseq
   pip install --editable ./
   cd ..
```
Don't forget to install other packages required for back translation,
```
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

pip install subword_nmt
pip install -U sacremoses==0.0.41
```

Git sacremoses and subword-nmt by,

```
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git
```

Since we need the script in sacremoses and subword-nmt for data preprocessing.

# A Walk Through

## Step1: Data Preprocessing

### Parallel Data ###

In the paper, we use the wmt14 en-de and en-ru langauge pairs, the language pairs are obtained from wmt14 website, here
we have a shell script for downloading and preprocess the data. Use DE-EN as an example.

the comman used for generate the parallel databin is in the shell script

```wmt14_en_de_prepare/start.sh```

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

### Monolingual Data ###

The command used for generate the monolingual databin is in the shell script

```prepare_wmt14/start.sh```

Also need to indicate several path such as the  

```$OUTPUT, $BPECODE ``` 

, etc.


## Step2: Baseline Training

we need to firstly train a baseline model. Take DE-EN as an example,

### Train de-en transformer-big model as baseline ###

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

### Train en-de transformer-big model as back-translation model ###

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
    $DE_EN_BT_BEAM_CHECKPOINT/checkpoint_best.pt
```

where, the ```$WMT14_DATABIN/code``` is the bpe code learned in bitext preprocess phrase.

## Step3: Back Transaltion

we use the back translation model to translate the target side languages back into source side

### Back Translation Synthetic Corpus ###

here we can use beam or sampling method, for beam generation, using

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

### Extract Back Translation Data ###

Then we can extract them out using default script in fairseq. It is at
fairseq/examples/backtranslation/extract_bt_data.py

```
python3 extract_bt_data.py \
    --minlen 1 --maxlen 512 \
    --output $PATH_TO_BEAM/extracted_bt_data --srclang de --tgtlang en \
    $PATH_TO_BEAM/beam5.out
```

and we preprocess this backtranslation data and combine the back translation data with bitext parallel databin

### Preprocess Back Translation Data ###

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

### Train Back Translation Forward Model ###

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

the command are in the script ```wmt14_de_en_bt_beam```

## Step4: Gamma

### Train Monolingual GPT Model ###

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

### Candidates Generation ###

for generate candidates, use

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

or, you can use shard to parallel sampling candidates, the preprocess shell will automatically generate the shard for
you.

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

where the ```$CANDIDATES_PATH``` is the path where the preprocess shell generate for each shard.

Then, we extract them out, and remove bpe code

```
for SHARD in $(seq -f "%02g" 0 29); do \
    python3 extract_bt_candidates.py \
        --output $CANDIDATES_PATH/shard${SHARD}/extracted_candidates.shard${SHARD} \
        --srclang de --tgtlang en  \
        < $CANDIDATES_PATH/shard${SHARD}/sampling50.shard${SHARD}.out
done
```

### Scoring Importance ###

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

## Step5: Gamma Corpus ##

there are two types of gamma corpus, deterministic(gamma selection) and stochastical method(gamma sampling).

### Gamma Selection ###

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

### Gamma Sampling ###

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

## Step5. Train Gamma Model ##

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

# Main Experiments
We conduct the experiments on WMT14 EN-DE and EN-RU datasets, the results are shown below.

| System          | EN-DE | DE-EN | EN-RU | RU-EN |
| --------------- | ----- | ----- | ----- | ----- |
| Transformer-big | 27.4  | 32.1  | 35.9  | 34.1  |
| Beam            | 29.7  | 32.7  | 39.6  | 35.9  |
| Sampling        | 30.0  | 34.1  | 37.4  | 35.6  |
| Gamma Selection | 31.0  | 34.7  | 35.7  | 36.1  |
| Gamma Sampling  | 30.9  | 35.0  | 38.9  | 36.3  |

# Datasets and Checkpoints
In this section, we list all the datasets and the checkpoint paths.
## Datasets:
### WMT14 EN-DE: ###
```/apdcephfs/share_916081/jettexu/importance_sampling/wmt14_en_de_full```
1. bitext databin: ```./databin```
2. EN side databin for GPT: ```./en_mono_databin```
3. DE side databin for GPT: ```./de_mono_databin```
4. EN monolingual dataset: ```./newscrawl_2020_en_4500000```
   1. mono databin: ```./newscrawl_2020_en_4500000/mono_databin```
   2. beam5: ```./newscrawl_2020_en_4500000/beam```
   3. sampling: ```./newscrawl_2020_en_4500000/sampling```
   4. 50 candidates: ```./newscrawl_2020_en_4500000/candidates```
      1. gamma selection: ```./newscrawl_2020_en_4500000/candidates/gamma0.2```
      2. gamma sampling: ```./newscrawl_2020_en_4500000/candidates/gamma_sampling0.2```
5. DE monolingual dataset: ```./newscrawl_2020_de_4500000```
   1. mono databin: ```./newscrawl_2020_de_4500000/mono_databin```
   2. beam5: ```./newscrawl_2020_de_4500000/beam```
   3. sampling: ```./newscrawl_2020_de_4500000/sampling```
   4. 50 candidates: ```./newscrawl_2020_de_4500000/candidates```
      1. gamma selection: ```./newscrawl_2020_de_4500000/candidates/gamma0.2```
      2. gamma sampling: ```./newscrawl_2020_de_4500000/candidates/gamma_sampling0.2```
   
### WMT14 EN-RU: ###
Path: ```/apdcephfs/share_916081/jettexu/importance_sampling/wmt14_en_ru```
1. bitext databin: ```./databin```
2. EN side databin for GPT: ```./en_mono_databin```
3. RU side databin for GPT: ```./ru_mono_databin```
4. EN monolingual dataset: ```./newscrawl_2020_en_2500000```
   1. mono databin: ```./newscrawl_2020_en_2500000/mono_databin```
   2. beam: ```./newscrawl_2020_en_2500000/beam```
   3. sampling: ```./newscrawl_2020_en_2500000/sampling```
   4. 50 candidaets: ```./newscrawl_2020_en_2500000/candidates```
      1. gamma selection: ```./newscrawl_2020_de_4500000/candidates/gamma0.2```
      2. gamma sampling: ```./newscrawl_2020_de_4500000/candidates/gamma_sampling0.2```
5. RU monolingual dataset: ```./newscrawl_2020_ru_2500000```
   1. mono databin: ```./newscrawl_2020_ru_2500000/mono_databin```
   2. beam: ```./newscrawl_2020_ru_2500000/beam```
   3. sampling: ```./newscrawl_2020_ru_2500000/sampling```
   4. 50 candidaets: ```./newscrawl_2020_ru_2500000/candidates```
      1. gamma selection: ```./newscrawl_2020_ru_4500000/candidates/gamma0.2```
      2. gamma sampling: ```./newscrawl_2020_ru_4500000/candidates/gamma_sampling0.2```

## Checkpoints:
Checkpoint path: ```/apdcephfs/share_916081/jettexu/importance_sampling/checkpoints/transformer_big```
Here we only list the gamma checkpoints at *gamma=0.2*
### WMT14 EN-DE: ###
1. baseline: ```./en_de_baseline```
2. beam: ```./en_de_bt_beam_mono_ratio1_lr7e-4_new_mono_002```
3. sampling: ```./en_de_bt_sampling_mono_ratio1_lr7e-4_new_mono_001```
4. gamma selection: ```./en_de_bt_sampling_mono_ratio1_lr7e-4_gamma0.2```
5. gamma sampling: ```./en_de_importance_random_sampling_mono_ratio1_gamma0.2```
6. EN gpt: ```./en_mono_gpt```

### WMT14 DE-EN: ###
1. baseline: ```./de_en_baseline```
2. beam: ```.de_en_bt_beam_mono_ratio1_lr7e-4_new_mono_002```
3. sampling: ```./de_en_bt_sampling_mono_ratio1_lr7e-4_new_mono_001```
4. gamma selection: ```./de_en_bt_sampling_mono_ratio1_lr7e-4_gamma0.2```
5. gamma sampling: ```./de_en_importance_random_sampling_mono_ratio1_gamma0.2```
6. DE gpt: ```./de_mono_gpt```

### WMT14 EN-RU ###
1. baseline: ```./en_ru_baseline_lr7e-4```
2. beam: ```./en_ru_bt_beam_mono_ratio1```
3. sampling: ```./en_ru_bt_sampling_mono_ratio1```
4. gamma selection: ```./en_ru_imp_sampling_mono_ratio1_gamma0.2```
5. gamma sampling: ```./en_ru_importance_random_sampling_mono_ratio1_gamma0.2```
6. EN gpt: ```./ru_en_mono_gpt_en```

### WMT14 RU-EN ###
1. baseline: ```./ru_en_baseline_lr7e-4```
2. beam: ```./ru_en_bt_beam```
3. sampling: ```./ru_en_bt_sampling```
4. gamma selection: ```./ru_en_imp_sampling_mono_ratio1_gamma0.2```
5. gamma sampling: ```./ru_en_importance_random_sampling_mono_ratio1_gamma0.2```
6. RU gpt: ```./ru_en_mono_gpt_ru```

# Analysis Experiment
## SVD Spectrum
for svd spectrum generation, one need to firstly save the sentence embedding bank,
We modify another version of fairseq named fairseq-main,
please change the code of ```fairseq-main/fairseq_cli/generate.py``` at ```line375```,
to modify the path you want to save the sentence representations. and use the 
```fairseq-generate``` command with ``` --score-reference``` flag to generate sentence embedding bank.
Under such a scenario, the source and target are all known to the model, and we just 
use the hidden state at end-of-speach token on last decoder as the sentence represenation.

once the embedding back is saved, one can use ```plot_embedding.py``` to plot the
t-SNE embedding and singular value spectrum plot. By indicating, the path to
sentence representation banks by dictionary,
```
paths={
        "beam":"/apdcephfs/share_916081/jettexu/importance_sampling/wmt14_en_de_full/databin/de-en/beam",
        "sampling":"/apdcephfs/share_916081/jettexu/importance_sampling/wmt14_en_de_full/databin/de-en/sampling",
        "gamma_selection":"/apdcephfs/share_916081/jettexu/importance_sampling/wmt14_en_de_full/databin/de-en/gamma_selection",
        "gamma_sampling":"/apdcephfs/share_916081/jettexu/importance_sampling/wmt14_en_de_full/databin/de-en/gamma_sampling"
    }
```

[comment]: <> (## Token Histogram)

[comment]: <> (For plotting the token level histogram, we use the ```plot.py``` .)

[comment]: <> (one need to firstly calculate the token frequency dictionary)





