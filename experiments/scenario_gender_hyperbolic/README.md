Run this code

```sh
PYTHONPATH="../../" python train_graph_embeddings.py device=${SC_DEVICE} --conf conf/conf.hocon

for SC_EPOCH in 001 002 003 004 005 006 007 008 009 010
do
    PYTHONPATH="../../" python inference.py \
        node_encoder="models/node_encoder.p" \
        nn_embeddings="models/nn_embedding_${SC_EPOCH}.p" \
        output="data/embedding_l2_${SC_EPOCH}.csv" \
        target_path="data/gender_train.csv"
done

rm results/check.txt 
# rm -r conf/check.work/
LUIGI_CONFIG_PATH="conf/luigi.cfg" \
    python -m embeddings_validation --conf conf/check.hocon \
    --workers 5 --total_cpu_count 18 \
    --conf_extra 'auto_features: ["../data/embedding_*00[1234].csv"]'
less -S results/check.txt

```

Todo:
- make batch bigger and train faster
