# run relation classification

# first create semeval dependency parses

train_file="../relation_classification_data/TRAIN_FILE.TXT"
train_outfile="semeval_train.tsv"
python create_dep_parses.py $train_file $train_outfile

test_file="../relation_classification_data/TEST_FILE_FULL.TXT"
test_outfile="semeval_test.tsv"
python create_dep_parses.py $test_file $test_outfile




# precompute ELMO embeddings

# generates semeval_elmo_embeddings_train_sentences.npy and semeval_elmo_embeddings_test_sentences.npy

CUDA_VISIBLE_DEVICES=0 python preprocessing_elmo.py



mkdir result_files
train_file="semeval10_meta_info_train.txt"
test_file="semeval10_meta_info_test.txt"
elmo_train="semeval_elmo_embeddings_train_sentences"
elmo_test="semeval_elmo_embeddings_test_sentences"


CUDA_VISIBLE_DEVICES=0 python training_elmo.py --labels_file=../relation_classification_data/labels.txt --train_file=$train_file --test_file=$test_file --word_embeddings_train=$elmo_train --word_embeddings_test=$elmo_test --num_epochs=25 --dropout=0.15 --batch_size=1000 --l2_lambda=0 --num_heads=2 --calculate_sdp_information=yes
