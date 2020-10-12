Run this code

```sh
# embedding_lr_0_001_500                           *** 0.542 0.016   0.526   0.559 0.013  [0.541 0.524 0.547 0.560 0.538]       0.512 0.009   0.503   0.522 0.008  [0.508 0.517 0.506 0.508 0.523]
export SC_SUFFIX="lr_0_001"   # this is new base
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    optimizer.lr=0.001 \
    --conf conf/conf.hocon

# embedding_lr_0_001_tree_batching_level_4_400     xxx 0.523 0.030   0.493   0.553 0.024  [0.509 0.501 0.507 0.547 0.550]       0.524 0.010   0.514   0.534 0.008  [0.515 0.524 0.534 0.530 0.517]
# batch_size=4 is bad option
export SC_SUFFIX="lr_0_001_tree_batching_level_4"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    optimizer.lr=0.001 \
    batch_size=4 tree_batching_level=4 \
    --conf conf/conf.hocon

# embedding_lr_0_001_topk_loss_0200                *** 0.547 0.004   0.543   0.552 0.003  [0.547 0.546 0.545 0.553 0.544]       0.534 0.026   0.508   0.560 0.021  [0.522 0.541 0.526 0.567 0.515]
### much better thah base ####
export SC_SUFFIX="lr_0_001_topk_loss"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=5 \
    loss.neg_mode="top_k" loss.k=32 loss.neg_margin=12.0 \
    --conf conf/conf.hocon

# embedding_lr_0_001_allneg_margin_12_0900         *** 0.547 0.014   0.533   0.562 0.012  [0.541 0.545 0.539 0.543 0.567]       0.531 0.005   0.527   0.536 0.004  [0.535 0.535 0.529 0.532 0.526]
# allneg better than top_k
export SC_SUFFIX="lr_0_001_allneg_margin_12"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=10 \
    loss.neg_margin=12.0 \
    --conf conf/conf.hocon

# embedding_lr_0_001_treelevel_1_0800              *** 0.553 0.022   0.531   0.575 0.018  [0.566 0.536 0.534 0.574 0.557]       0.536 0.010   0.526   0.546 0.008  [0.528 0.531 0.541 0.532 0.547]
export SC_SUFFIX="lr_0_001_treelevel_1"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=20 epoch_n=2000 \
    tree_batching_level=1 batch_size=64 \
    --conf conf/conf.hocon

# this is new base
# embedding_lr_0_001_treelevel_1_margin_12_1000    *** 0.552 0.011   0.541   0.563 0.009  [0.542 0.559 0.561 0.553 0.544]       0.554 0.020   0.535   0.574 0.016  [0.536 0.560 0.548 0.579 0.548]
#     Epoch [1000]: node_count: 344.922, pos_count: 1404.793, neg_count: 13525.598, loss: 135464.256: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 496/496 [00:04<00:00, 102.72it/s]
#     validate               : Validation: 210998 pos samples 190212 neg samples. Pos hit: 0.94637                                                                                                                                                                                          
#     validate               : Distances:                                                                                                                                                                                                                                                   
#                           pos    neg  pos_pp  neg_pp                                                                                                                                                                                                                                      
#       0.759 -   2.185     168     42    0.00    0.00                                                                                                                                                                                                                                      
#       2.185 -   3.612    1014     23    0.00    0.00                                                                                                                                                                                                                                      
#       3.612 -   5.038    3682    125    0.02    0.00                                                                                                                                                                                                                                      
#       5.038 -   6.464   14430    426    0.07    0.00                                                                                                                                                                                                                                      
#       6.464 -   7.891   47366   3750    0.22    0.02                                                                                                                                                                                                                                      
#       7.891 -   9.317  104608  28016    0.50    0.15                                                                                                                                                                                                                                      
#       9.317 -  10.743   38126  92314    0.18    0.49                                                                                                                                                                                                                                      
#      10.743 -  12.170    1598  61151    0.01    0.32                                                                                                                                                                                                                                      
#      12.170 -  13.596       6   4365    0.00    0.02                                                                                                                                                                                                                                      
#     
export SC_SUFFIX="lr_0_001_treelevel_1_margin_12"
PYTHONPATH="../../" python train_graph_embeddings.py \
    device=${SC_DEVICE} \
    model_prefix="models/${SC_SUFFIX}_" \
    valid.epoch_step=20 epoch_n=1000 \
    tree_batching_level=1 batch_size=64 \
    loss.neg_margin=12.0 \
    --conf conf/conf.hocon


####################################################
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

