export SC_SUFFIX="l2_128_old_base"   # this is new base
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device="${SC_DEVICE}" \
    model_prefix="models/${SC_SUFFIX}/" \
    norm_embedding_weights=false \
    loss.pos_margin=0.1 loss.neg_margin=16 \
    log_file.path="models/${SC_SUFFIX}/log_model_${SC_SUFFIX}.json" log_file.feature_name="embeddings_${SC_SUFFIX}" \
    --conf conf/conf.hocon


export SC_SUFFIX="l2_128_base"   # this is new base
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device="${SC_DEVICE}" \
    model_prefix="models/${SC_SUFFIX}/" \
    log_file.path="models/${SC_SUFFIX}/log_model_${SC_SUFFIX}.json" log_file.feature_name="embeddings_${SC_SUFFIX}" \
    --conf conf/conf.hocon


export SC_SUFFIX="l2_128_pos"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device="${SC_DEVICE}" \
    loss.pos_margin=0.1 \
    model_prefix="models/${SC_SUFFIX}/" \
    log_file.path="models/${SC_SUFFIX}/log_model_${SC_SUFFIX}.json" log_file.feature_name="embeddings_${SC_SUFFIX}" \
    --conf conf/conf.hocon




####################################################
####### Get embeddings and validate
#
#for SC_SUFFIX in "l2_128_base_1" "l2_128_base_2" "l2_128_pos"
#do
#  for SC_EPOCH in 0001 0005 0010 0020 0040 0080 0100 0120 0150 0200 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500 1600
#  do
#      PYTHONPATH="../../" python inference.py \
#          node_encoder="models/${SC_SUFFIX}/node_encoder.p" \
#          nn_embeddings="models/${SC_SUFFIX}/nn_embedding_${SC_EPOCH}.p" \
#          output="data/embedding_${SC_SUFFIX}_${SC_EPOCH}.csv" \
#          target_path="data/gender_train.csv"
#  done
#done
rm results/check_train.txt
rm -r conf/check_train.work/
LUIGI_CONFIG_PATH="conf/luigi.cfg" \
    python -m embeddings_validation --conf conf/check_train.hocon \
    --workers 5 --total_cpu_count 18
less -S results/check_train.txt
#
