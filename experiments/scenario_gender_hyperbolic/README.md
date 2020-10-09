Run this code

# base 0.02215 after 10 epochs

```sh
export SC_SUFFIX="base"

export SC_SUFFIX="lr_0_001"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    optimizer.lr=0.001 \
    --conf conf/conf.hocon

export SC_SUFFIX="lr_0_001_tree_batching_level_4"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    optimizer.lr=0.001 \
    batch_size=4 tree_batching_level=4 \
    --conf conf/conf.hocon



export SC_SUFFIX="base"
for SC_EPOCH in 001 010 100 200 300 400 500 600 700 800 900 1000
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
 
