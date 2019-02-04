# relation_extraction

# preprocessing semeval
first precompute elmo embeddings

labels_file: a file with all the labels in semeval data, one label per line
train_infile: path to the official semeval10 train file ("TRAIN_FILE.TXT")
test_infile: path to the official semeval10 testfile ("TEST_FILE_FULL.TXT")
train_outfile: the outfile generated by the preprocessing (with the relevant information for the dataset)
test_outfile: the same for test
path_to_word_embedding_train: path to which file the word embeddings should be saved
path_to_test_embeddings: same for test

python preprocessing_elmo.py --labels_file path_to_labels_file --train_infile path_to_train --test_infile path_to_test --train_outfile path_to_outfile_train --test_outfile path_to_outfile_test --word_embeddings_train path_to_word_embedding_train --word_embeddings_test path_to_word_embeddings_test

# run the model
then run the model, for our final version: run with

CUDA_VISIBLE_DEVICES=0 python training_elmo.py --labels_file=../relation_classification_data/labels.txt --train_file=$train_file --test_file=$test_file --word_embeddings_train=$elmo_train --word_embeddings_test=$elmo_test --num_epochs=25 --dropout=0.15 --batch_size=1000 --l2_lambda=0 --num_heads=2 --calculate_sdp_information=yes

# run full pipeline

simply run 

bash run_full_rc_pipeline.sh

# dependencies

pip install spacy

python -m spacy download en

pip install numpy 

pip install nltk

python -c 'import nltk; nltk.download("stopwords");'

pip install tensorflow-gpu

pip install tensorflow-hub
