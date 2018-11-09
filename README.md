# relation_extraction

first precompute elmo embeddings

python preprocessing_elmo.py

then run the model, for our final version: run with

python training_elmo.py --encoder_attention no --decoder_self_attention no --encoder_feedforward no --resultfile final_results_semveval_
