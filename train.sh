set -x

train=$1
devel=$2
modeldir=$3

python OpenNMT-py/preprocess.py -train_src $train.input -train_tgt $train.output -valid_src $devel.input -valid_tgt $devel.output -save_data $modeldir/inflection

python OpenNMT-py/train.py -data $modeldir/inflection -save_model $modeldir/inflection-model -gpuid 0 -dropout 0.1
