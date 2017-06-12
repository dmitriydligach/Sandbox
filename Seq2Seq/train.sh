export MODEL_DIR=/home/dima/Temp/Model
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      /Users/Dima/Git/Sandbox/Seq2Seq/nmt_small.yml,
      /Users/Dima/Git/Sandbox/Seq2Seq/train_seq2seq.yml,
      /Users/Dima/Git/Sandbox/Seq2Seq/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
