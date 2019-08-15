#!bin/bash

module load python/3.7.0
module load pytorch/1.1.0

IMPORTS=(
    filtered-iitb.tar
    ilci.tar
    national-newscrawl.tar
    ufal-en-tam.tar
    wat-ilmpc.tar
)

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"


mkdir -p $LOCAL_ROOT/{data,checkpoints}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/ufal-transformer-big/checkpoints

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}

# copy
export ILMULTI_CORPUS_ROOT=$DATA

set -x
function train {
    python3 train.py \
        --task shared-multilingual-translation \
        --num-workers 0 \
        --arch transformer \
        --max-tokens 5000 --lr 1e-4 --min-lr 1e-9 \
        --optimizer adam \
        --save-dir $CHECKPOINTS \
        --log-format simple --log-interval 200 \
        --criterion label_smoothed_cross_entropy \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
        --ddp-backend no_c10d \
        --update-freq 2 \
        config.yaml 
        # --reset-optimizer \
}

function _test {
    python3 generate.py config.yaml \
        --task shared-multilingual-translation  \
        --path $CHECKPOINTS/checkpoint_last.pt > ufal-gen.out
    cat ufal-gen.out \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > ufal-test.hyp
    cat ufal-gen.out \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > ufal-test.ref

    split -d -l 2000 ufal-test.hyp hyp.ufal.
    split -d -l 2000 ufal-test.ref ref.ufal.

    # perl multi-bleu.perl ref.ufal.00 < hyp.ufal.00 
    # perl multi-bleu.perl ref.ufal.01 < hyp.ufal.01 

    python3 -m indicnlp.contrib.wat.evaluate \
        --reference ref.ufal.00 --hypothesis hyp.ufal.00 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference ref.ufal.01 --hypothesis hyp.ufal.01 

}

ARG=$1
eval "$1"
# _test
