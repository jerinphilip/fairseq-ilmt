#!/bin/bash
#SBATCH --job-name=pib
#SBATCH --partition short
#SBATCH --account shashanks
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time 06:00:00
##SBATCH -w gnode01
##SBATCH --reservation non-deadline-queue

module add use.own
module load python/3.7.0
module load pytorch/1.1.0
<<CMT
IMPORTS=(
    mann-ki-baat-test.tar
)

IMPORTS=(
    odiencorp.tar
)
CMT
LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"


mkdir -p $LOCAL_ROOT/{data,checkpoints}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints

#rsync -rvz /home/shashanks/ilci/ $DATA/ilci/
#rsync -rvz ada:/share1/shashanks/checkpoints/pib_ilci/checkpoint_last.pt $CHECKPOINTS/checkpoint_last.pt



#rsync -rvz /home/shashanks/mkb/ $DATA/mkb/
#rsync -rvz ada:/share1/shashanks/checkpoints/pib/checkpoint_best.pt $CHECKPOINTS/checkpoint_last.pt

#rsync -rvz ada:/share1/shashanks/checkpoints/pib_ilci/checkpoint_last.pt $CHECKPOINTS/checkpoint_last.pt



###out of domain eval
#rsync -rvz /home/shashanks/mkb/ $DATA/mkb/
#rsync -rvz ada:/share1/shashanks/checkpoints/pib/checkpoint_best.pt $CHECKPOINTS/checkpoint_last.pt
#rsync -rvz ada:/share1/shashanks/checkpoints/pib_ilci/checkpoint_last.pt $CHECKPOINTS/checkpoint_last.pt


function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}

#copy

export ILMULTI_CORPUS_ROOT=$DATA

python3 preprocess_cvit.py test_config.yaml

OUT=/home/shashanks/fairseq-cvit/out/ilci+pib-ilci_domain

set -x


function _test {

    python3 generate.py test_config.yaml \
        --task shared-multilingual-translation  \
        --path $CHECKPOINTS/checkpoint_last.pt > $OUT/pib_m0.out
<<CMT
    cat $OUT/pib_m0.out \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_m0.hyp
    cat $OUT/pib_m0.out \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_m0.ref
    cat $OUT/pib_m0.out \
        | grep "^S" | sed 's/^S-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_m0.src
CMT

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
