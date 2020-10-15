# embedding_lr_0_001_500                           *** 0.542 0.016   0.526   0.559 0.013  [0.541 0.524 0.547 0.560 0.538]       0.512 0.009   0.503   0.522 0.008  [0.508 0.517 0.506 0.508 0.523]
export SC_SUFFIX="base"   # this is new base
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_128"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=128 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_256_sigma"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=256 loss.neg_margin=18.6 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_256_mf"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=256 loss.neg_margin=22.6 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_256_mf_lr"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=256 loss.neg_margin=22.6 optimizer.lr=0.0014 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon


export SC_SUFFIX="base_128_mf"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=128 loss.neg_margin=16 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_064"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=64 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_032"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=32 loss.neg_margin=4.3 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_032_mf"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=32 loss.neg_margin=8 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon

export SC_SUFFIX="base_032_sigma"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    embedding_dim=32 loss.neg_margin=2.8 \
    valid.epoch_step=50 \
    --conf conf/conf.hocon


export SC_SUFFIX="poincare"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="poincare" loss.neg_margin=6.0 valid.batch_size=40 \
    --conf conf/conf.hocon


export SC_SUFFIX="poincare_wnorm"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=10 \
    norm_embedding_weights=true \
    optimizer.lr=0.001 \
    distance="poincare" loss.neg_margin=1.0 valid.batch_size=40 \
    --conf conf/conf.hocon



export SC_SUFFIX="sphere"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="sphere" loss.neg_margin=1.0 valid.batch_size=40 \
    optimizer.lr=0.01 \
    --conf conf/conf.hocon

export SC_SUFFIX="sphere_normw_2"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="sphere" loss.neg_margin=1.0 valid.batch_size=40 \
    norm_embedding_weights=true \
    optimizer.lr=0.01 \
    --conf conf/conf.hocon



export SC_SUFFIX="hyperbola"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="hyperbola" loss.neg_margin=6.0 valid.batch_size=40 \
    --conf conf/conf.hocon

export SC_SUFFIX="sphere_1.4"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="sphere" loss.neg_margin=1.4 valid.batch_size=40 \
    optimizer.lr=0.02 \
    --conf conf/conf.hocon

export SC_SUFFIX="dotcos"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="dotcos" loss.neg_margin=0.9 valid.batch_size=40 \
    optimizer.lr=0.01 \
    --conf conf/conf.hocon

export SC_SUFFIX="acos"
mkdir "models/${SC_SUFFIX}/"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}/" \
    valid.epoch_step=20 \
    distance="acos" loss.neg_margin=0.6 valid.batch_size=40 \
    optimizer.lr=0.01 \
    --conf conf/conf.hocon


####################################################
####### Get embeddings and validate

#
for SC_SUFFIX in "poincare_wnorm"
do
  # for SC_EPOCH in 0001 0005 0010 0020 0040 0080 0100 0120 0200 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500 1600
  for SC_EPOCH in 0100
  do
      PYTHONPATH="../../" python inference.py \
          node_encoder="models/${SC_SUFFIX}/node_encoder.p" \
          nn_embeddings="models/${SC_SUFFIX}/nn_embedding_${SC_EPOCH}.p" \
          output="data/embedding_${SC_SUFFIX}_${SC_EPOCH}.csv" \
          target_path="data/gender_train.csv"
  done
done

rm results/check.txt 
# rm -r conf/check.work/
LUIGI_CONFIG_PATH="conf/luigi.cfg" \
    python -m embeddings_validation --conf conf/check.hocon \
    --workers 5 --total_cpu_count 18 \
    --conf_extra 'auto_features: ["../data/embedding_*.csv"]'
less -S results/check.txt
