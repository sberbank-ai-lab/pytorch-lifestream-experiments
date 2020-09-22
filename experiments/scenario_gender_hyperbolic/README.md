Run this code

```sh
export SC_SUFFIX="base"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    --conf conf/conf.hocon

for SC_EPOCH in 001 002 003 004 005 006 007 008 009 010
do
    PYTHONPATH="../../" python inference.py \
        node_encoder="models/${SC_SUFFIX}_node_encoder.p" \
        nn_embeddings="models/${SC_SUFFIX}_nn_embedding_${SC_EPOCH}.p" \
        output="data/embedding_${SC_SUFFIX}_${SC_EPOCH}.csv" \
        target_path="data/gender_train.csv"
done

rm results/check.txt 
rm -r conf/check.work/
LUIGI_CONFIG_PATH="conf/luigi.cfg" \
    python -m embeddings_validation --conf conf/check.hocon \
    --workers 5 --total_cpu_count 18 \
    --conf_extra 'auto_features: ["../data/embedding_*.csv"]'
less -S results/check.txt

```

Todo:
- make batch bigger and train faster
