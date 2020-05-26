#!/bin/bash
#SBATCH --job-name=pib-iter1-branch-toEN
#SBATCH --partition short
#SBATCH --account shashanks
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time 0-06:00:00
#SBATCH --signal=B:HUP@1000
#SBATCH -w gnode41

module add use.own
module load python/3.7.0
module load pytorch/1.1.0


IMPORTS=(
    ilci-split.tar
    filtered-iitb.tar
    national-newscrawl.tar
    wat-ilmpc.tar
    ufal-en-tam.tar
    bible-en-te.tar
    eenadu-en-te.tar
    odiencorp.tar
    pib-v0.tar
)

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
FSEQ=/home/shashanks/fairseq-cvit
PIB=$FSEQ/pib-exps
CONFIG=$PIB/pib-train-config.yaml

#rm -r $LOCAL_ROOT/{data,checkpoints}

mkdir -p $LOCAL_ROOT/{data,checkpoints}

set -x

rsync -rvz ada:/share3/shashanks/checkpoints/pib_all/iter1/toEN/checkpoint_last.pt $CHECKPOINTS/checkpoint_last.pt

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}


function _export {
    ssh $USER@ada "mkdir -p ada:/share3/shashanks/checkpoints/pib_all/iter1/toEN/"
    rsync -rvz $CHECKPOINTS/*.pt ada:/share3/$USER/checkpoints/pib_all/iter1/toEN/
}

trap "_export" SIGHUP
export ILMULTI_CORPUS_ROOT=$DATA

#copy

#python3 $FSEQ/preprocess_cvit.py $CONFIG

ARCH='transformer'
MAX_TOKENS=3500
LR=1e-3
UPDATE_FREQ=1
MAX_EPOCHS=50

python3 $FSEQ/train.py \
    --num-workers 0 \
    --task shared-multilingual-translation \
    --arch $ARCH --save-dir $CHECKPOINTS \
    --share-all-embeddings \
    --encoder-normalize-before --decoder-normalize-before \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
    --label-smoothing 0.2 --criterion label_smoothed_cross_entropy\
    --max-tokens $MAX_TOKENS --lr $LR --min-lr 1e-9 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --log-format simple --log-interval 200 \
    --lr-scheduler inverse_sqrt \
    --clip-norm 0 --weight-decay 0.0001 \
    --ddp-backend no_c10d \
    --update-freq $UPDATE_FREQ \
    --max-epoch $MAX_EPOCHS \
    --save-interval-updates 20000 \
    $CONFIG &

wait

_export

# --save-interval-updates 10000 \
# --dropout 0.4 --attention-dropout 0.2 --activation-dropout 0.2 \
    




    
