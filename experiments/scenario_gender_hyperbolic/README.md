Run this code

```sh
PYTHONPATH="../../" python train_graph_embeddings.py device="cuda:1" --conf conf/conf.hocon

for SC_EPOCH in 005 009
do
    PYTHONPATH="../../" python inference.py \
        node_encoder="models/node_encoder.p" \
        nn_embeddings="models/nn_embedding_${SC_EPOCH}.p" \
        output="data/embedding_l2_${SC_EPOCH}.csv" \
        target_path="data/gender_train.csv"
done

python -m embeddings_validation --conf conf/check.hocon --workers 5 --total_cpu_count 12

```

Todo:
- make batch bigger and train faster
