# export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u main.py \
  --do_train True \
  --epoch 30 \
  --src_vocab_fpath data/wmt18_zhen_data_bpe3/vocab_all.bpe.32000 \
  --trg_vocab_fpath data/wmt18_zhen_data_bpe3/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file data/wmt18_zhen_data_bpe3/train.tok.clean.bpe.32000.en-zh \
  --batch_size 4096