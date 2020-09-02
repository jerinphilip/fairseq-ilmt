#!/bin/bash
#SBATCH --job-name=bt
#SBATCH --partition long
#SBATCH --account shashanks
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time 3-00:00:00
#SBATCH --signal=B:HUP@600
##SBATCH -w gnode37


module add use.own
module load python/3.7.0
module load pytorch/1.1.0

<<CMT
IMPORTS=(
    filtered-iitb.tar
    ilci.tar
    national-newscrawl.tar
    ufal-en-tam.tar
    wat-ilmpc.tar
    bible-en-te.tar
    eenadu-en-te.tar
)
CMT

IMPORTS=(
    odiencorp.tar
)

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"

mkdir -p $LOCAL_ROOT/{data,checkpoints}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints

rsync -r /home/shashanks/ilci/ $DATA/ilci/

#rsync -rvz ada:/share1/shashanks/checkpoints/mmall/checkpoint_last.pt $CHECKPOINTS/

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}

function _export {
    ssh $USER@ada "mkdir -p ada:/share1/$USER/checkpoints/pib"
    rsync -rvz $CHECKPOINTS/checkpoint_{best,last}.pt ada:/share1/$USER/checkpoints/pib/
}

trap "_export" SIGHUP
copy
export ILMULTI_CORPUS_ROOT=$DATA

python3 preprocess_cvit.py config.yaml


ARCH='transformer'
MAX_TOKENS=3500
LR=1e-3
UPDATE_FREQ=128
MAX_EPOCHS=200

set -x
function train {
    python3 train.py \
        --task shared-multilingual-translation \
        --share-all-embeddings \
        --num-workers 0 \
        --arch $ARCH \
        --max-tokens $MAX_TOKENS --lr $LR --min-lr 1e-9 \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --save-dir $CHECKPOINTS \
        --log-format simple --log-interval 200 \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
        --lr-scheduler inverse_sqrt \
        --clip-norm 0.1 \
        --ddp-backend no_c10d \
        --update-freq $UPDATE_FREQ \
        --max-epoch $MAX_EPOCHS \
        --criterion label_smoothed_cross_entropy \
        config.yaml

}

    #    --reset-optimizer \
    #    --reset-lr-scheduler \

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

wait
_export
