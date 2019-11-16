#!/bin/bash
#SBATCH --job-name=pib
#SBATCH --partition short
#SBATCH --account shashanks
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time 06:00:00
#SBATCH --reservation non-deadline-queue
##SBATCH -w gnode01

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


mkdir -p $LOCAL_ROOT/{data,checkpoints}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
FSEQ=/home/shashanks/fairseq-cvit
PIB=$FSEQ/pib-exps
OUT=$PIB/output/iter0

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}
copy

rsync -rvz /home/shashanks/ilci/ $DATA/ilci/
rsync -rvz ada:/share1/shashanks/checkpoints/pib_all/checkpoint_best.pt $CHECKPOINTS/checkpoint_last.pt

set -x

export ILMULTI_CORPUS_ROOT=$DATA

python3 $FSEQ/preprocess_cvit.py $PIB/test-config.yaml

function _test {

    python3 $FSEQ/generate.py $PIB/test-config.yaml \
        --task shared-multilingual-translation  \
        --skip-invalid-size-inputs-valid-test \
        --path $CHECKPOINTS/checkpoint_last.pt > $OUT/pib_m0.out

    cat $OUT/pib_m0.out \
        | grep "^H" | cut -f 3 \
            > $OUT/pib_m0.ind
    cat $OUT/pib_m0.out \
        | grep "^H" | cut -f 5 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_m0.hyp
    cat $OUT/pib_m0.out \
        | grep "^T" | cut -f 4 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_m0.ref

}

_test
