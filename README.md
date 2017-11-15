# inflect-words
Neural model for word inflection using OpenNMT and pytorch libraries.

Needs python3 environment with pytorch installed.

## Prepering data from .conllu files

    python prepare_data.py -f train.conllu -o data/train
    python prepare_data.py -f devel.conllu -o data/devel
    
    python prepare_data.py -f devel.conllu -o test_data/test
    
Creates data/train.input and data/train.output files (and same for devel and test files).

## Train neural inflection model with OpenNMT

    ./train.sh data/train data/devel model_directory
    
Creates trained model under model_directory.

## Predict test data

    python predict.py -model model_name -src test_data/test.input -output pred.txt -gpu 0
    
Predicted inflections are now in the pred.txt file.

## Test agains gold standard inflections

    cat test_data/test.output | perl -pe 's/ //g' > test_data/test.gold
    python accuracy.py -g test_data/test.gold -s pred.txt
    
Optionally, if you include original input file for test set (test_data/test.input) with --original_input option, it also reports accuracy for each part-of-speech category.
