Run this code

```sh
export SC_SUFFIX="lr_0_001"   # this is new base
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    optimizer.lr=0.001 \
    --conf conf/conf.hocon

# done-break
export SC_SUFFIX="lr_0_001_tree_batching_level_4"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    optimizer.lr=0.001 \
    batch_size=4 tree_batching_level=4 \
    --conf conf/conf.hocon

# cuda:0
export SC_SUFFIX="lr_0_001_topk_loss"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    loss.neg_mode="top_k" loss.k=32 loss.neg_margin=12.0 \
    --conf conf/conf.hocon

# done
export SC_SUFFIX="lr_0_001_treelevel_1"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=20 epoch_n=2000 \
    tree_batching_level=1 batch_size=64 \
    --conf conf/conf.hocon

# cuda:1
# fast result on 100 epoch
export SC_SUFFIX="lr_0_001_treelevel_1_margin_12"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=20 epoch_n=1000 \
    tree_batching_level=1 batch_size=64 \
    loss.neg_margin=12.0 \
    --conf conf/conf.hocon

# cuda:2
export SC_SUFFIX="lr_0_001_allneg_margin_12"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=10 \
    loss.neg_margin=12.0 \
    --conf conf/conf.hocon



####### Get embeddings and validate

export SC_SUFFIX="lr_0_001_treelevel_1_margin_12"
for SC_EPOCH in 0001 0010 0100 0200 0300 0400 0500 0600 0700 0800 0900 1000
do
    PYTHONPATH="../../" python inference.py \
        node_encoder="models/${SC_SUFFIX}_node_encoder.p" \
        nn_embeddings="models/${SC_SUFFIX}_nn_embedding_${SC_EPOCH}.p" \
        output="data/embedding_${SC_SUFFIX}_${SC_EPOCH}.csv" \
        target_path="data/gender_train.csv"
done

rm results/check.txt 
# rm -r conf/check.work/
LUIGI_CONFIG_PATH="conf/luigi.cfg" \
    python -m embeddings_validation --conf conf/check.hocon \
    --workers 5 --total_cpu_count 18 \
    --conf_extra 'auto_features: ["../data/embedding_*.csv"]'
less -S results/check.txt

```

