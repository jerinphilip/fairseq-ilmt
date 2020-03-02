#!/bin/bash
#SBATCH --job-name=pib
#SBATCH --partition long
#SBATCH --account shashanks
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time 1-00:00:00
##SBATCH --reservation non-deadline-queue
##SBATCH -w gnode40

module add use.own
module load python/3.7.0
module load pytorch/1.1.0

IMPORTS=(
    filtered-iitb.tar
    wat-ilmpc.tar
    ilci-split.tar
    ufal-en-tam.tar
    bible-en-te.tar
    mkb-v0.tar
    odiencorp.tar
)

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"


DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
FSEQ=/home/shashanks/fairseq-cvit
PIB=$FSEQ/pib-exps

CONFIG=$PIB/test-config.yaml
OUT=$PIB/output/cold/new_multi/best

rm -r $LOCAL_ROOT/{data,checkpoints}

mkdir -p $LOCAL_ROOT/{data,checkpoints}
mkdir -p $OUT

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}
copy

#rsync -rvz /home/shashanks/pib-test/ $DATA/pib-test/

rsync -rvz ada:/share1/shashanks/checkpoints/pib_all/cold/new_multi/checkpoint_best.pt $CHECKPOINTS/checkpoint_last.pt


set -x

export ILMULTI_CORPUS_ROOT=$DATA

python3 $FSEQ/preprocess_cvit.py $CONFIG


MODEL=pib

function _test {

    python3 $FSEQ/generate.py $CONFIG \
       --task shared-multilingual-translation  \
       --skip-invalid-size-inputs-valid-test \
       --path $CHECKPOINTS/checkpoint_last.pt > $OUT/pib_$MODEL.out

    cat $OUT/pib_$MODEL.out \
        | grep "^H" | cut -f 3 \
            > $OUT/pib_$MODEL.ind
    cat $OUT/pib_$MODEL.out \
        | grep "^H" | cut -f 5 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_$MODEL.hyp
    cat $OUT/pib_$MODEL.out \
        | grep "^T" | cut -f 4 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_$MODEL.ref

}

_test

    