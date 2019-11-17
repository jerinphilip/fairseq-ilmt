#!/bin/bash
#SBATCH --job-name=pib-train-all
#SBATCH --partition long
#SBATCH --account shashanks
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=45G
#SBATCH --time 3-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH --reservation non-deadline-queue
#SBATCH -w gnode18

module add use.own
module load python/3.7.0
module load pytorch/1.1.0


IMPORTS=(
    filtered-iitb.tar
    #ilci.tar
    national-newscrawl.tar
    ufal-en-tam.tar
    wat-ilmpc.tar
    bible-en-te.tar
    eenadu-en-te.tar
    odiencorp.tar
)


LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"


DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
FSEQ=/home/shashanks/fairseq-cvit
PIB=$FSEQ/pib-exps

mkdir -p $LOCAL_ROOT/{data,checkpoints}

#rsync -rvz /home/shashanks/ilci/ $DATA/ilci/
##rsync -rvz ada:/share1/shashanks/checkpoints/pib_all/checkpoint_last.pt $CHECKPOINTS/checkpoint_last.pt

set -x

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}


function _export {
    ssh $USER@ada "mkdir -p ada:/share1/shashanks/checkpoints/pib_all"
    rsync -rvz $CHECKPOINTS/*.pt ada:/share1/$USER/checkpoints/pib_all/
}

trap "_export" SIGHUP
export ILMULTI_CORPUS_ROOT=$DATA

#copy

python3 $FSEQ/preprocess_cvit.py $PIB/pib-train-config.yaml


ARCH='transformer'
MAX_TOKENS=3500
LR=1e-3
UPDATE_FREQ=1
MAX_EPOCHS=50


python3 $FSEQ/train.py \
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
    --save-interval-updates 1000 \
    --reset-optimizer \
    --reset-lr-scheduler \
    $PIB/pib-train-config.yaml &

wait

_export