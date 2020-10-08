Run this code

```sh
export SC_SUFFIX="l2_tree_batching_level_1"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    epoch_n=100 \
    tree_batching_level=1 batch_size=96 \
    distance="l2" \
    model_prefix="models/${SC_SUFFIX}_" \
    --conf conf/conf.hocon

for SC_EPOCH in 001 010 020 030 040 050 060 070 080 090 100
do
    PYTHONPATH="../../" python inference.py \
        node_encoder="models/${SC_SUFFIX}_node_encoder.p" \
        nn_embeddings="models/${SC_SUFFIX}_nn_embedding_${SC_EPOCH}.p" \
        output="data/embedding_${SC_SUFFIX}_${SC_EPOCH}.csv" \
        target_path="data/gender_train.csv"
done


export SC_SUFFIX="l2_tree_batching_level_3"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    epoch_n=100 \
    tree_batching_level=3 batch_size=4 \
    distance="l2" \
    model_prefix="models/${SC_SUFFIX}_" \
    --conf conf/conf.hocon
export SC_SUFFIX="l2_tree_batching_level_3"

for SC_EPOCH in 001 010 020 030 040 050 060 070 080 090 100
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

Todo:
- make batch bigger and train faster


 0  02:02  90%
 1  01:57  92%
 
