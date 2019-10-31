#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition long
#SBATCH --account shashanks
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time 06:00:00

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

set -x

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="ada:/share1/dataset/text"
SHARE="ada:/share3/$USER"

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
FSEQ=/home/$USER/fairseq-cvit
PIB=$FSEQ/pib-exps
CONFIG=$PIB/test-config.yaml
ITER=iter3
OUT=$PIB/output/generation-results/$ITER/to-en/reported

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

rsync -rvz /home/shashanks/ilci/ $DATA/ilci/
copy

rsync -rvz $SHARE/newstest2019guen/ $DATA/newstest2019guen/
rsync -rvz $SHARE/newstest2019engu/ $DATA/newstest2019engu/


rsync -rvz $SHARE/checkpoints/pib_all/iter3/toEN/checkpoint_last.pt $CHECKPOINTS/checkpoint_last.pt

export ILMULTI_CORPUS_ROOT=$DATA

python3 $FSEQ/preprocess_cvit.py $CONFIG

MODEL=$ITER

function _test {

    python3 $FSEQ/generate.py $CONFIG \
       --task shared-multilingual-translation  \
       --skip-invalid-size-inputs-valid-test \
       --path $CHECKPOINTS/checkpoint_last.pt > $OUT/pib_$MODEL.out

    cat $OUT/pib_$MODEL.out \
        | grep "^H" | sort -nk 2  | cut -f 3 \
            > $OUT/pib_$MODEL.ind
    cat $OUT/pib_$MODEL.out \
        | grep "^H" | sort -nk 2  | cut -f 5 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_$MODEL.hyp
    cat $OUT/pib_$MODEL.out \
        | grep "^T" | sort -nk 2  | cut -f 4 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > $OUT/pib_$MODEL.ref

    python3 $FSEQ/generate_cvit.py \
        --ind $OUT/pib_$MODEL.ind \
        --hyp $OUT/pib_$MODEL.hyp \
        --ref $OUT/pib_$MODEL.ref \
        --out_dir $OUT \
        --test_config $CONFIG \

}

_test
