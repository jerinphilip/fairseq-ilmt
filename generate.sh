#!/bin/bash
#SBATCH --job-name=pib
#SBATCH --partition long
#SBATCH --account shashanks
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time 1-00:00:00
##SBATCH -w gnode22
#SBATCH --reservation non-deadline-queue

module add use.own
module load python/3.7.0
module load pytorch/1.1.0
<<CMT
IMPORTS=(
    mann-ki-baat-test.tar
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

#rsync -r /home/shashanks/ilci/ $DATA/ilci/
#rsync -rvz ada:/share1/shashanks/checkpoints/pib/checkpoint_best.pt $CHECKPOINTS/checkpoint_last.pt

function copy {
    for IMPORT in ${IMPORTS[@]}; do
        rsync --progress $REMOTE_ROOT/$IMPORT $DATA/
        tar_args="$DATA/$IMPORT -C $DATA/"
        tar -df $tar_args 2> /dev/null || tar -kxvf $tar_args
    done
}

#copy
export ILMULTI_CORPUS_ROOT=$DATA

OUT=/home/shashanks/fairseq-cvit/out/out2

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

    split -d -l 500 $OUT/pib_m0.hyp $OUT/pib_m0.hyp.
    split -d -l 500 $OUT/pib_m0.ref $OUT/pib_m0.ref.
    split -d -l 500 $OUT/pib_m0.src $OUT/pib_m0.src.


    # perl multi-bleu.perl ref.ufal.00 < hyp.ufal.00 
    # perl multi-bleu.perl ref.ufal.01 < hyp.ufal.01 
    echo "en<-->hi"
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/pib_m0.ref.00 --hypothesis $OUT/pib_m0.hyp.00 > $OUT/pib_m0.hyp.00.res.txt
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/pib_m0.ref.01 --hypothesis $OUT/pib_m0.hyp.01 > $OUT/pib_m0.hyp.01.res.txt
    echo "en<-->ta"
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.02 --hypothesis $OUT/hyp.mkb.02 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.03 --hypothesis $OUT/hyp.mkb.03
    echo "en<-->te"
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.04 --hypothesis $OUT/hyp.mkb.04 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.05 --hypothesis $OUT/hyp.mkb.05
    echo "hi<-->ta"
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.06 --hypothesis $OUT/hyp.mkb.06 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.07 --hypothesis $OUT/hyp.mkb.07
    echo "hi<-->te"
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.08 --hypothesis $OUT/hyp.mkb.08 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.09 --hypothesis $OUT/hyp.mkb.09
    echo "ta<-->te"
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.10 --hypothesis $OUT/hyp.mkb.10 
    python3 -m indicnlp.contrib.wat.evaluate \
        --reference $OUT/ref.mkb.11 --hypothesis $OUT/hyp.mkb.01
CMT
}

_test
