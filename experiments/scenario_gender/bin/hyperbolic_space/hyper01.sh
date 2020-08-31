export SC_SUFFIX="hyper01_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=0.5 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper13_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=0.5 \
    params.rnn.trainable_starter="static" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="static" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper14_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=0.5 \
    params.rnn.trainable_starter="none" \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper16_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=0.5 \
    params.rnn.trainable_starter="none" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper17_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper18_base"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss   params.train.margin=0.5 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="l2_atan" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="l2_atan" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper09_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper19_hyperbolic"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="hyperbolic" params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="atan" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="atan" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper20_hyperbolic"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="euclidean" params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="atan" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="atan" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper22_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.rnn.hidden_size=16 \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.hidden_size=16 \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

export SC_SUFFIX="hyper12_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 \
    params.rnn.trainable_starter="static" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="static" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done


export SC_SUFFIX="hyper23_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="exp" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

python -m scenario_gender compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_hyper23*.pickle"


export SC_SUFFIX="hyper24_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 params.train.l2_w=0.0001 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

python -m scenario_gender compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_hyper24*.pickle"

export SC_SUFFIX="hyper25_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 params.train.l2_w=0.001 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

python -m scenario_gender compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_hyper25*.pickle"

export SC_SUFFIX="hyper29_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 params.train.l2_w=10.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

python -m scenario_gender compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_hyper29*.pickle"

export SC_SUFFIX="hyper28_poincare"
python ../../metric_learning.py \
    params.device="$SC_DEVICE" \
    params.train.loss=ContrastiveLoss params.train.distance="poincare" params.train.margin=2.0 params.train.l2_w=1.0 \
    params.rnn.trainable_starter="none" \
    params.use_normalization_layer="poincare" \
    params.train.lr=0.004 \
    model_path.model="models/mles_model_${SC_SUFFIX}.p" \
    params.train.checkpoints.save_interval=10 \
    params.train.checkpoints.n_saved=1000 \
    params.train.checkpoints.dirname="models/mles_checkpoints_${SC_SUFFIX}/" \
    params.train.checkpoints.filename_prefix="mles" \
    params.train.checkpoints.create_dir=true \
    --conf conf/dataset.hocon conf/mles_params.json
for SC_EPOCH in 010 020 030 040 050 060 070 080 090 100 110 120 130 140 150
do
    python ../../ml_inference.py \
        params.rnn.trainable_starter="none" \
        params.use_normalization_layer="poincare" \
        model_path.model="models/mles_checkpoints_${SC_SUFFIX}/mles_model_${SC_EPOCH##+(0)}.pt" \
        output.path="data/mles_${SC_SUFFIX}_${SC_EPOCH}" \
        params.device="$SC_DEVICE" --conf conf/dataset.hocon conf/mles_params.json
done

python -m scenario_gender compare_approaches --n_workers 3 --models lgb \
    --embedding_file_names "mles_hyper28*.pickle"


                                         oof_rocauc_score                                                         test_rocauc_score
                                                     mean t_int_l t_int_h    std                           values              mean t_int_l t_int_h    std                           values
name
lgb_embeds: mles_hyper01_base_010.pickle           0.8556  0.8463  0.8648 0.0075  [0.844 0.852 0.859 0.860 0.862]            0.8621  0.8585  0.8657 0.0029  [0.859 0.860 0.862 0.864 0.865]
lgb_embeds: mles_hyper01_base_020.pickle           0.8611  0.8505  0.8718 0.0085  [0.850 0.854 0.865 0.868 0.869]            0.8687  0.8647  0.8727 0.0032  [0.864 0.868 0.868 0.869 0.873]
lgb_embeds: mles_hyper01_base_030.pickle           0.8643  0.8569  0.8717 0.0059  [0.858 0.858 0.867 0.867 0.871]            0.8681  0.8627  0.8735 0.0043  [0.861 0.868 0.868 0.870 0.873]
lgb_embeds: mles_hyper01_base_040.pickle           0.8629  0.8553  0.8706 0.0061  [0.855 0.859 0.864 0.866 0.871]            0.8710  0.8691  0.8728 0.0015  [0.869 0.870 0.871 0.871 0.873]
lgb_embeds: mles_hyper01_base_050.pickle           0.8665  0.8551  0.8778 0.0092  [0.854 0.860 0.868 0.874 0.876]            0.8778  0.8753  0.8803 0.0020  [0.875 0.877 0.877 0.878 0.881]
lgb_embeds: mles_hyper01_base_060.pickle           0.8668  0.8588  0.8749 0.0065  [0.859 0.863 0.865 0.873 0.874]            0.8762  0.8731  0.8794 0.0025  [0.873 0.875 0.877 0.877 0.879]
lgb_embeds: mles_hyper01_base_070.pickle           0.8679  0.8597  0.8762 0.0066  [0.859 0.862 0.871 0.873 0.874]            0.8820  0.8790  0.8849 0.0024  [0.879 0.880 0.883 0.883 0.885]
lgb_embeds: mles_hyper01_base_080.pickle           0.8694  0.8615  0.8772 0.0063  [0.860 0.867 0.872 0.874 0.875]            0.8775  0.8740  0.8809 0.0028  [0.874 0.876 0.877 0.879 0.881]
lgb_embeds: mles_hyper01_base_090.pickle           0.8697  0.8607  0.8788 0.0073  [0.862 0.862 0.871 0.875 0.878]            0.8781  0.8771  0.8791 0.0008  [0.877 0.878 0.878 0.879 0.879]
lgb_embeds: mles_hyper01_base_100.pickle           0.8674  0.8593  0.8754 0.0065  [0.859 0.864 0.866 0.873 0.875]            0.8791  0.8768  0.8814 0.0019  [0.877 0.877 0.880 0.880 0.881]
lgb_embeds: mles_hyper01_base_110.pickle           0.8697  0.8607  0.8786 0.0072  [0.862 0.864 0.868 0.876 0.878]            0.8788  0.8754  0.8823 0.0028  [0.877 0.878 0.878 0.878 0.884]
lgb_embeds: mles_hyper01_base_120.pickle           0.8682  0.8596  0.8768 0.0069  [0.860 0.862 0.871 0.873 0.875]            0.8769  0.8735  0.8802 0.0027  [0.875 0.876 0.876 0.876 0.882]
lgb_embeds: mles_hyper01_base_130.pickle           0.8688  0.8619  0.8756 0.0055  [0.862 0.864 0.870 0.873 0.875]            0.8789  0.8748  0.8830 0.0033  [0.873 0.880 0.880 0.880 0.881]
lgb_embeds: mles_hyper01_base_140.pickle       *** 0.8690  0.8604  0.8776 0.0069  [0.861 0.863 0.869 0.876 0.876]            0.8813  0.8765  0.8860 0.0038  [0.878 0.879 0.881 0.881 0.888]
lgb_embeds: mles_hyper01_base_150.pickle           0.8689  0.8602  0.8775 0.0070  [0.862 0.862 0.868 0.876 0.876]            0.8812  0.8768  0.8857 0.0036  [0.876 0.879 0.882 0.884 0.885]

lgb_embeds: mles_hyper13_base_010.pickle           0.8542  0.8431  0.8654 0.0090  [0.841 0.848 0.859 0.860 0.862]            0.8595  0.8560  0.8630 0.0028  [0.856 0.859 0.859 0.860 0.864]
lgb_embeds: mles_hyper13_base_020.pickle           0.8578  0.8480  0.8675 0.0079  [0.846 0.854 0.859 0.864 0.866]            0.8640  0.8606  0.8673 0.0027  [0.861 0.862 0.863 0.866 0.867]
lgb_embeds: mles_hyper13_base_030.pickle           0.8588  0.8491  0.8685 0.0078  [0.848 0.852 0.864 0.864 0.866]            0.8692  0.8663  0.8721 0.0023  [0.867 0.868 0.868 0.869 0.873]
lgb_embeds: mles_hyper13_base_040.pickle           0.8603  0.8502  0.8704 0.0082  [0.852 0.852 0.859 0.869 0.869]            0.8683  0.8653  0.8714 0.0024  [0.866 0.867 0.867 0.871 0.871]
lgb_embeds: mles_hyper13_base_050.pickle           0.8635  0.8518  0.8752 0.0094  [0.851 0.857 0.866 0.872 0.872]            0.8777  0.8762  0.8793 0.0012  [0.876 0.877 0.877 0.879 0.879]
lgb_embeds: mles_hyper13_base_060.pickle           0.8664  0.8546  0.8783 0.0095  [0.853 0.862 0.867 0.875 0.876]            0.8772  0.8757  0.8787 0.0012  [0.876 0.876 0.877 0.878 0.879]
lgb_embeds: mles_hyper13_base_070.pickle           0.8661  0.8566  0.8756 0.0076  [0.857 0.859 0.867 0.871 0.875]            0.8792  0.8753  0.8830 0.0031  [0.876 0.876 0.881 0.882 0.882]
lgb_embeds: mles_hyper13_base_080.pickle           0.8662  0.8538  0.8787 0.0101  [0.855 0.858 0.866 0.874 0.879]            0.8794  0.8778  0.8810 0.0013  [0.878 0.878 0.880 0.881 0.881]
lgb_embeds: mles_hyper13_base_090.pickle           0.8666  0.8542  0.8791 0.0100  [0.855 0.858 0.868 0.874 0.879]            0.8818  0.8795  0.8841 0.0019  [0.880 0.880 0.882 0.883 0.884]
lgb_embeds: mles_hyper13_base_100.pickle           0.8676  0.8546  0.8807 0.0105  [0.855 0.861 0.866 0.877 0.880]            0.8808  0.8772  0.8844 0.0029  [0.878 0.879 0.879 0.883 0.885]
lgb_embeds: mles_hyper13_base_110.pickle           0.8674  0.8543  0.8805 0.0105  [0.853 0.860 0.872 0.873 0.879]            0.8807  0.8775  0.8839 0.0026  [0.878 0.878 0.882 0.882 0.883]
lgb_embeds: mles_hyper13_base_120.pickle           0.8673  0.8543  0.8803 0.0105  [0.855 0.857 0.872 0.874 0.878]            0.8855  0.8832  0.8877 0.0018  [0.883 0.885 0.885 0.887 0.888]
lgb_embeds: mles_hyper13_base_130.pickle           0.8656  0.8543  0.8769 0.0091  [0.854 0.860 0.866 0.873 0.875]            0.8824  0.8790  0.8859 0.0028  [0.881 0.881 0.881 0.883 0.887]
lgb_embeds: mles_hyper13_base_140.pickle           0.8664  0.8551  0.8777 0.0091  [0.854 0.860 0.868 0.874 0.875]            0.8815  0.8792  0.8837 0.0018  [0.880 0.881 0.881 0.881 0.885]
lgb_embeds: mles_hyper13_base_150.pickle           0.8655  0.8533  0.8777 0.0098  [0.855 0.855 0.870 0.872 0.876]            0.8811  0.8770  0.8853 0.0033  [0.878 0.879 0.880 0.882 0.887]

lgb_embeds: mles_hyper14_base_010.pickle           0.8540  0.8442  0.8639 0.0079  [0.844 0.847 0.858 0.860 0.861]            0.8519  0.8489  0.8549 0.0024  [0.850 0.850 0.851 0.854 0.855]
lgb_embeds: mles_hyper14_base_020.pickle           0.8606  0.8525  0.8686 0.0065  [0.851 0.856 0.863 0.865 0.867]            0.8696  0.8663  0.8729 0.0027  [0.868 0.868 0.868 0.871 0.873]
lgb_embeds: mles_hyper14_base_030.pickle           0.8660  0.8545  0.8776 0.0093  [0.854 0.861 0.865 0.872 0.878]            0.8763  0.8715  0.8810 0.0039  [0.872 0.874 0.875 0.877 0.882]
lgb_embeds: mles_hyper14_base_040.pickle           0.8684  0.8600  0.8769 0.0068  [0.859 0.866 0.868 0.874 0.876]            0.8749  0.8716  0.8781 0.0026  [0.873 0.873 0.874 0.874 0.879]
lgb_embeds: mles_hyper14_base_050.pickle           0.8704  0.8608  0.8800 0.0077  [0.858 0.869 0.872 0.876 0.877]            0.8782  0.8748  0.8815 0.0027  [0.874 0.877 0.879 0.879 0.881]
lgb_embeds: mles_hyper14_base_060.pickle           0.8709  0.8636  0.8783 0.0060  [0.863 0.868 0.870 0.875 0.879]            0.8784  0.8737  0.8831 0.0038  [0.875 0.876 0.877 0.880 0.884]
lgb_embeds: mles_hyper14_base_070.pickle           0.8728  0.8669  0.8787 0.0048  [0.867 0.870 0.872 0.877 0.879]            0.8811  0.8789  0.8833 0.0017  [0.879 0.880 0.881 0.881 0.884]
lgb_embeds: mles_hyper14_base_080.pickle           0.8717  0.8644  0.8789 0.0058  [0.865 0.868 0.869 0.877 0.879]            0.8757  0.8737  0.8777 0.0016  [0.874 0.875 0.876 0.876 0.878]
lgb_embeds: mles_hyper14_base_090.pickle           0.8710  0.8638  0.8782 0.0058  [0.864 0.866 0.872 0.875 0.878]            0.8735  0.8698  0.8771 0.0030  [0.871 0.871 0.873 0.874 0.878]
lgb_embeds: mles_hyper14_base_100.pickle           0.8712  0.8636  0.8787 0.0061  [0.865 0.865 0.873 0.875 0.878]            0.8741  0.8703  0.8778 0.0030  [0.869 0.873 0.874 0.876 0.877]
lgb_embeds: mles_hyper14_base_110.pickle           0.8715  0.8633  0.8798 0.0067  [0.861 0.869 0.873 0.876 0.879]            0.8738  0.8724  0.8752 0.0011  [0.873 0.873 0.874 0.874 0.876]
lgb_embeds: mles_hyper14_base_120.pickle           0.8709  0.8626  0.8792 0.0067  [0.863 0.867 0.869 0.878 0.878]            0.8730  0.8692  0.8768 0.0030  [0.871 0.872 0.872 0.872 0.878]
lgb_embeds: mles_hyper14_base_130.pickle           0.8705  0.8611  0.8798 0.0075  [0.860 0.867 0.871 0.875 0.879]            0.8731  0.8705  0.8757 0.0021  [0.871 0.872 0.873 0.874 0.876]
lgb_embeds: mles_hyper14_base_140.pickle           0.8714  0.8635  0.8794 0.0064  [0.863 0.869 0.870 0.877 0.878]            0.8766  0.8736  0.8796 0.0024  [0.873 0.876 0.876 0.878 0.879]
lgb_embeds: mles_hyper14_base_150.pickle           0.8711  0.8633  0.8789 0.0063  [0.863 0.868 0.869 0.876 0.879]            0.8724  0.8686  0.8762 0.0031  [0.869 0.870 0.872 0.875 0.876]

lgb_embeds: mles_hyper15_base_010.pickle           0.8314  0.8211  0.8417 0.0083  [0.821 0.827 0.829 0.839 0.841]            0.8419  0.8378  0.8459 0.0033  [0.837 0.841 0.842 0.845 0.845]
lgb_embeds: mles_hyper15_base_020.pickle           0.8317  0.8169  0.8464 0.0119  [0.817 0.824 0.830 0.843 0.844]            0.8445  0.8385  0.8505 0.0048  [0.839 0.841 0.845 0.847 0.851]
lgb_embeds: mles_hyper15_base_030.pickle           0.8411  0.8300  0.8521 0.0089  [0.829 0.835 0.844 0.846 0.851]            0.8504  0.8467  0.8540 0.0030  [0.845 0.851 0.852 0.852 0.852]
lgb_embeds: mles_hyper15_base_040.pickle           0.8460  0.8343  0.8578 0.0094  [0.834 0.837 0.851 0.852 0.855]            0.8531  0.8461  0.8602 0.0057  [0.848 0.848 0.852 0.857 0.861]
lgb_embeds: mles_hyper15_base_050.pickle           0.8484  0.8388  0.8579 0.0077  [0.840 0.841 0.852 0.852 0.857]            0.8529  0.8482  0.8577 0.0038  [0.849 0.850 0.852 0.855 0.858]
lgb_embeds: mles_hyper15_base_060.pickle           0.8507  0.8419  0.8594 0.0070  [0.840 0.847 0.854 0.855 0.857]            0.8567  0.8550  0.8584 0.0014  [0.855 0.856 0.857 0.858 0.858]
lgb_embeds: mles_hyper15_base_070.pickle           0.8503  0.8398  0.8608 0.0085  [0.840 0.844 0.852 0.857 0.860]            0.8576  0.8511  0.8640 0.0052  [0.853 0.854 0.854 0.863 0.863]
lgb_embeds: mles_hyper15_base_080.pickle           0.8495  0.8389  0.8601 0.0085  [0.838 0.843 0.852 0.856 0.858]            0.8561  0.8511  0.8611 0.0041  [0.852 0.853 0.855 0.859 0.862]
lgb_embeds: mles_hyper15_base_090.pickle           0.8533  0.8436  0.8630 0.0078  [0.844 0.846 0.857 0.858 0.862]            0.8583  0.8550  0.8616 0.0027  [0.854 0.858 0.859 0.859 0.862]
lgb_embeds: mles_hyper15_base_100.pickle           0.8522  0.8437  0.8607 0.0068  [0.842 0.848 0.857 0.857 0.857]            0.8584  0.8533  0.8634 0.0041  [0.853 0.856 0.859 0.860 0.864]
lgb_embeds: mles_hyper15_base_110.pickle           0.8528  0.8422  0.8633 0.0085  [0.842 0.845 0.859 0.859 0.860]            0.8576  0.8541  0.8610 0.0028  [0.854 0.857 0.858 0.858 0.861]
lgb_embeds: mles_hyper15_base_120.pickle           0.8555  0.8450  0.8660 0.0084  [0.843 0.852 0.858 0.860 0.865]            0.8594  0.8570  0.8618 0.0019  [0.857 0.859 0.860 0.860 0.862]
lgb_embeds: mles_hyper15_base_130.pickle           0.8566  0.8449  0.8683 0.0094  [0.845 0.848 0.862 0.863 0.865]            0.8618  0.8591  0.8644 0.0022  [0.859 0.860 0.862 0.862 0.865]
lgb_embeds: mles_hyper15_base_140.pickle           0.8557  0.8468  0.8647 0.0072  [0.847 0.849 0.857 0.861 0.864]            0.8617  0.8580  0.8654 0.0030  [0.859 0.860 0.861 0.862 0.867]
lgb_embeds: mles_hyper15_base_150.pickle           0.8580  0.8478  0.8683 0.0083  [0.846 0.853 0.862 0.864 0.864]            0.8622  0.8585  0.8659 0.0030  [0.858 0.861 0.862 0.863 0.867]

lgb_embeds: mles_hyper16_base_010.pickle           0.8592  0.8485  0.8699 0.0086  [0.844 0.859 0.863 0.865 0.865]            0.8692  0.8657  0.8727 0.0028  [0.865 0.868 0.870 0.870 0.873]
lgb_embeds: mles_hyper16_base_020.pickle           0.8639  0.8535  0.8743 0.0084  [0.852 0.861 0.863 0.870 0.873]            0.8745  0.8730  0.8761 0.0013  [0.873 0.874 0.875 0.875 0.876]
lgb_embeds: mles_hyper16_base_030.pickle           0.8681  0.8572  0.8791 0.0088  [0.858 0.859 0.872 0.875 0.876]            0.8744  0.8698  0.8789 0.0037  [0.871 0.873 0.874 0.874 0.880]
lgb_embeds: mles_hyper16_base_040.pickle           0.8699  0.8607  0.8792 0.0075  [0.859 0.866 0.873 0.876 0.876]            0.8747  0.8714  0.8779 0.0026  [0.872 0.872 0.875 0.877 0.878]
lgb_embeds: mles_hyper16_base_050.pickle           0.8689  0.8602  0.8776 0.0070  [0.861 0.862 0.873 0.874 0.875]            0.8719  0.8671  0.8767 0.0039  [0.867 0.871 0.871 0.872 0.878]
lgb_embeds: mles_hyper16_base_060.pickle           0.8682  0.8591  0.8772 0.0073  [0.858 0.864 0.869 0.874 0.875]            0.8753  0.8713  0.8794 0.0032  [0.872 0.873 0.875 0.877 0.880]
lgb_embeds: mles_hyper16_base_070.pickle           0.8709  0.8627  0.8790 0.0065  [0.863 0.865 0.874 0.875 0.877]            0.8790  0.8745  0.8834 0.0036  [0.877 0.877 0.878 0.879 0.885]
lgb_embeds: mles_hyper16_base_080.pickle           0.8704  0.8602  0.8805 0.0082  [0.861 0.863 0.872 0.877 0.879]            0.8765  0.8704  0.8826 0.0049  [0.873 0.873 0.874 0.878 0.885]
lgb_embeds: mles_hyper16_base_090.pickle           0.8688  0.8602  0.8774 0.0069  [0.860 0.864 0.868 0.875 0.876]            0.8703  0.8671  0.8736 0.0026  [0.867 0.868 0.871 0.872 0.874]
lgb_embeds: mles_hyper16_base_100.pickle           0.8695  0.8600  0.8790 0.0076  [0.860 0.866 0.867 0.877 0.877]            0.8737  0.8708  0.8766 0.0023  [0.871 0.873 0.873 0.875 0.877]
lgb_embeds: mles_hyper16_base_110.pickle           0.8686  0.8579  0.8792 0.0086  [0.858 0.861 0.871 0.876 0.877]            0.8738  0.8712  0.8764 0.0021  [0.873 0.873 0.873 0.873 0.878]
lgb_embeds: mles_hyper16_base_120.pickle           0.8695  0.8585  0.8805 0.0088  [0.857 0.864 0.871 0.877 0.879]            0.8742  0.8703  0.8781 0.0031  [0.869 0.874 0.875 0.876 0.877]
lgb_embeds: mles_hyper16_base_130.pickle           0.8698  0.8607  0.8790 0.0074  [0.863 0.863 0.869 0.877 0.878]            0.8730  0.8699  0.8762 0.0025  [0.870 0.872 0.872 0.875 0.876]
lgb_embeds: mles_hyper16_base_140.pickle           0.8701  0.8608  0.8793 0.0075  [0.860 0.864 0.872 0.875 0.878]            0.8754  0.8709  0.8799 0.0036  [0.871 0.873 0.875 0.878 0.880]
lgb_embeds: mles_hyper16_base_150.pickle           0.8679  0.8600  0.8759 0.0064  [0.859 0.866 0.867 0.873 0.874]            0.8719  0.8689  0.8750 0.0025  [0.869 0.870 0.871 0.874 0.875]

lgb_embeds: mles_hyper18_base_010.pickle           0.8601  0.8503  0.8699 0.0079  [0.851 0.853 0.864 0.866 0.868]            0.8712  0.8680  0.8745 0.0026  [0.868 0.870 0.871 0.872 0.875]
lgb_embeds: mles_hyper18_base_020.pickle           0.8663  0.8554  0.8773 0.0088  [0.853 0.862 0.868 0.873 0.875]            0.8716  0.8671  0.8761 0.0036  [0.867 0.868 0.872 0.874 0.876]
lgb_embeds: mles_hyper18_base_030.pickle           0.8682  0.8575  0.8789 0.0086  [0.858 0.863 0.866 0.877 0.877]            0.8767  0.8739  0.8796 0.0023  [0.873 0.876 0.877 0.879 0.879]
lgb_embeds: mles_hyper18_base_040.pickle           0.8683  0.8582  0.8784 0.0081  [0.857 0.864 0.869 0.875 0.877]            0.8739  0.8696  0.8782 0.0035  [0.870 0.871 0.874 0.877 0.878]
lgb_embeds: mles_hyper18_base_050.pickle           0.8714  0.8610  0.8819 0.0084  [0.859 0.869 0.871 0.878 0.880]            0.8765  0.8750  0.8779 0.0012  [0.875 0.876 0.876 0.876 0.878]
lgb_embeds: mles_hyper18_base_060.pickle           0.8715  0.8598  0.8833 0.0095  [0.860 0.865 0.870 0.880 0.882]            0.8765  0.8737  0.8793 0.0023  [0.874 0.875 0.877 0.878 0.879]
lgb_embeds: mles_hyper18_base_070.pickle           0.8700  0.8597  0.8804 0.0083  [0.858 0.868 0.868 0.877 0.879]            0.8762  0.8732  0.8793 0.0024  [0.873 0.876 0.876 0.877 0.880]
lgb_embeds: mles_hyper18_base_080.pickle           0.8711  0.8613  0.8809 0.0079  [0.862 0.867 0.868 0.879 0.880]            0.8783  0.8764  0.8802 0.0015  [0.877 0.877 0.878 0.879 0.880]
lgb_embeds: mles_hyper18_base_090.pickle           0.8702  0.8620  0.8785 0.0066  [0.863 0.866 0.869 0.873 0.880]            0.8717  0.8689  0.8744 0.0022  [0.869 0.869 0.872 0.873 0.874]
lgb_embeds: mles_hyper18_base_100.pickle           0.8693  0.8590  0.8796 0.0083  [0.860 0.865 0.866 0.878 0.878]            0.8710  0.8677  0.8743 0.0026  [0.869 0.869 0.871 0.871 0.875]
lgb_embeds: mles_hyper18_base_110.pickle           0.8685  0.8561  0.8809 0.0100  [0.855 0.864 0.866 0.878 0.879]            0.8743  0.8722  0.8763 0.0016  [0.872 0.873 0.875 0.876 0.876]
lgb_embeds: mles_hyper18_base_120.pickle           0.8702  0.8598  0.8805 0.0083  [0.860 0.865 0.868 0.878 0.880]            0.8735  0.8690  0.8780 0.0036  [0.871 0.871 0.872 0.874 0.880]
lgb_embeds: mles_hyper18_base_130.pickle           0.8705  0.8586  0.8824 0.0096  [0.860 0.864 0.868 0.877 0.883]            0.8747  0.8728  0.8766 0.0016  [0.873 0.873 0.875 0.875 0.877]
lgb_embeds: mles_hyper18_base_140.pickle           0.8710  0.8599  0.8821 0.0090  [0.860 0.867 0.868 0.878 0.882]            0.8731  0.8679  0.8783 0.0042  [0.867 0.870 0.875 0.876 0.878]
lgb_embeds: mles_hyper18_base_150.pickle           0.8707  0.8609  0.8804 0.0078  [0.861 0.867 0.869 0.877 0.880]            0.8740  0.8695  0.8785 0.0036  [0.870 0.872 0.873 0.877 0.878]

lgb_embeds: mles_hyper09_poincare_010.pickle           0.8598  0.8513  0.8684 0.0069  [0.852 0.853 0.862 0.865 0.867]            0.8707  0.8678  0.8737 0.0024  [0.867 0.871 0.871 0.871 0.874]
lgb_embeds: mles_hyper09_poincare_020.pickle           0.8624  0.8538  0.8710 0.0069  [0.854 0.856 0.865 0.867 0.870]            0.8719  0.8677  0.8762 0.0034  [0.869 0.870 0.870 0.874 0.877]
lgb_embeds: mles_hyper09_poincare_030.pickle           0.8694  0.8582  0.8805 0.0090  [0.857 0.864 0.871 0.874 0.880]            0.8833  0.8793  0.8873 0.0032  [0.878 0.882 0.883 0.886 0.886]
lgb_embeds: mles_hyper09_poincare_040.pickle           0.8715  0.8593  0.8836 0.0098  [0.862 0.863 0.868 0.881 0.883]            0.8835  0.8797  0.8873 0.0030  [0.879 0.882 0.884 0.886 0.887]
lgb_embeds: mles_hyper09_poincare_050.pickle           0.8719  0.8642  0.8796 0.0062  [0.865 0.866 0.872 0.876 0.880]            0.8832  0.8809  0.8854 0.0018  [0.881 0.882 0.883 0.884 0.885]
lgb_embeds: mles_hyper09_poincare_060.pickle           0.8727  0.8627  0.8826 0.0080  [0.862 0.867 0.874 0.880 0.881]            0.8839  0.8806  0.8872 0.0027  [0.882 0.882 0.883 0.883 0.889]
lgb_embeds: mles_hyper09_poincare_070.pickle           0.8726  0.8627  0.8826 0.0080  [0.865 0.866 0.870 0.880 0.882]            0.8846  0.8812  0.8881 0.0028  [0.881 0.883 0.885 0.886 0.888]
lgb_embeds: mles_hyper09_poincare_080.pickle           0.8710  0.8605  0.8815 0.0084  [0.861 0.864 0.872 0.877 0.881]            0.8838  0.8810  0.8866 0.0023  [0.882 0.882 0.884 0.884 0.887]
lgb_embeds: mles_hyper09_poincare_090.pickle           0.8713  0.8641  0.8785 0.0058  [0.865 0.867 0.871 0.873 0.880]            0.8799  0.8761  0.8837 0.0030  [0.877 0.877 0.880 0.881 0.884]
lgb_embeds: mles_hyper09_poincare_100.pickle           0.8703  0.8612  0.8794 0.0073  [0.863 0.867 0.867 0.874 0.881]            0.8873  0.8831  0.8916 0.0034  [0.884 0.885 0.887 0.889 0.892]
lgb_embeds: mles_hyper09_poincare_110.pickle           0.8728  0.8629  0.8827 0.0080  [0.865 0.866 0.871 0.881 0.881]            0.8835  0.8797  0.8874 0.0031  [0.880 0.882 0.883 0.885 0.888]
lgb_embeds: mles_hyper09_poincare_120.pickle       *** 0.8730  0.8633  0.8826 0.0077  [0.865 0.867 0.871 0.881 0.881]            0.8816  0.8788  0.8844 0.0023  [0.878 0.881 0.882 0.883 0.884]
lgb_embeds: mles_hyper09_poincare_130.pickle           0.8720  0.8634  0.8806 0.0070  [0.865 0.866 0.870 0.878 0.881]            0.8802  0.8769  0.8836 0.0027  [0.878 0.878 0.879 0.881 0.884]
lgb_embeds: mles_hyper09_poincare_140.pickle           0.8720  0.8617  0.8822 0.0083  [0.863 0.866 0.871 0.878 0.882]            0.8811  0.8771  0.8851 0.0032  [0.877 0.879 0.881 0.884 0.884]
lgb_embeds: mles_hyper09_poincare_150.pickle           0.8716  0.8611  0.8822 0.0085  [0.863 0.864 0.872 0.879 0.881]            0.8790  0.8768  0.8812 0.0018  [0.877 0.878 0.880 0.881 0.881]

lgb_embeds: mles_hyper19_hyperbolic_010.pickle           0.8573  0.8458  0.8689 0.0093  [0.845 0.851 0.862 0.863 0.866]            0.8704  0.8666  0.8743 0.0031  [0.868 0.869 0.870 0.871 0.875]
lgb_embeds: mles_hyper19_hyperbolic_020.pickle           0.8595  0.8495  0.8695 0.0081  [0.851 0.852 0.862 0.865 0.869]            0.8777  0.8739  0.8815 0.0031  [0.874 0.875 0.878 0.880 0.882]
lgb_embeds: mles_hyper19_hyperbolic_030.pickle           0.8662  0.8540  0.8783 0.0098  [0.852 0.860 0.870 0.874 0.875]            0.8848  0.8797  0.8898 0.0041  [0.881 0.884 0.884 0.885 0.892]
lgb_embeds: mles_hyper19_hyperbolic_040.pickle           0.8663  0.8561  0.8765 0.0082  [0.853 0.863 0.870 0.872 0.873]            0.8792  0.8746  0.8838 0.0037  [0.876 0.877 0.878 0.880 0.885]
lgb_embeds: mles_hyper19_hyperbolic_050.pickle           0.8691  0.8594  0.8788 0.0078  [0.858 0.865 0.874 0.875 0.875]            0.8815  0.8765  0.8866 0.0041  [0.877 0.880 0.880 0.882 0.888]
lgb_embeds: mles_hyper19_hyperbolic_060.pickle           0.8682  0.8579  0.8786 0.0083  [0.856 0.866 0.867 0.875 0.877]            0.8786  0.8751  0.8821 0.0028  [0.875 0.877 0.879 0.880 0.882]
lgb_embeds: mles_hyper19_hyperbolic_070.pickle           0.8688  0.8591  0.8784 0.0078  [0.856 0.867 0.870 0.874 0.876]            0.8821  0.8763  0.8880 0.0047  [0.878 0.880 0.880 0.882 0.890]
lgb_embeds: mles_hyper19_hyperbolic_080.pickle           0.8665  0.8551  0.8779 0.0091  [0.853 0.865 0.867 0.870 0.878]            0.8812  0.8797  0.8827 0.0012  [0.880 0.881 0.881 0.882 0.882]
lgb_embeds: mles_hyper19_hyperbolic_090.pickle           0.8666  0.8580  0.8753 0.0070  [0.855 0.866 0.867 0.870 0.874]            0.8771  0.8734  0.8807 0.0029  [0.874 0.876 0.877 0.877 0.882]
lgb_embeds: mles_hyper19_hyperbolic_100.pickle           0.8665  0.8549  0.8781 0.0094  [0.852 0.865 0.867 0.874 0.875]            0.8759  0.8719  0.8798 0.0032  [0.873 0.874 0.874 0.876 0.881]
lgb_embeds: mles_hyper19_hyperbolic_110.pickle           0.8667  0.8566  0.8768 0.0081  [0.852 0.868 0.870 0.872 0.872]            0.8806  0.8753  0.8859 0.0043  [0.875 0.878 0.880 0.883 0.886]
lgb_embeds: mles_hyper19_hyperbolic_120.pickle           0.8682  0.8585  0.8779 0.0078  [0.856 0.867 0.868 0.873 0.877]            0.8781  0.8727  0.8836 0.0044  [0.873 0.876 0.878 0.879 0.885]
lgb_embeds: mles_hyper19_hyperbolic_130.pickle           0.8678  0.8602  0.8754 0.0061  [0.858 0.866 0.869 0.871 0.874]            0.8786  0.8760  0.8813 0.0021  [0.877 0.877 0.878 0.880 0.882]
lgb_embeds: mles_hyper19_hyperbolic_140.pickle           0.8660  0.8559  0.8761 0.0081  [0.852 0.867 0.867 0.871 0.873]            0.8782  0.8722  0.8841 0.0048  [0.873 0.875 0.879 0.880 0.885]
lgb_embeds: mles_hyper19_hyperbolic_150.pickle           0.8669  0.8565  0.8772 0.0083  [0.854 0.865 0.868 0.873 0.874]            0.8762  0.8704  0.8821 0.0047  [0.872 0.874 0.875 0.877 0.884]

lgb_embeds: mles_hyper20_hyperbolic_010.pickle           0.8562  0.8473  0.8652 0.0072  [0.844 0.856 0.859 0.859 0.863]            0.8657  0.8631  0.8684 0.0021  [0.863 0.865 0.865 0.866 0.869]
lgb_embeds: mles_hyper20_hyperbolic_020.pickle           0.8595  0.8460  0.8730 0.0109  [0.846 0.852 0.859 0.868 0.873]            0.8666  0.8618  0.8714 0.0039  [0.863 0.863 0.867 0.868 0.872]
lgb_embeds: mles_hyper20_hyperbolic_030.pickle           0.8651  0.8531  0.8772 0.0097  [0.852 0.860 0.864 0.874 0.875]            0.8786  0.8747  0.8824 0.0031  [0.876 0.876 0.878 0.879 0.884]
lgb_embeds: mles_hyper20_hyperbolic_040.pickle           0.8688  0.8574  0.8802 0.0092  [0.854 0.867 0.869 0.876 0.877]            0.8787  0.8742  0.8833 0.0037  [0.875 0.876 0.877 0.880 0.884]
lgb_embeds: mles_hyper20_hyperbolic_050.pickle           0.8676  0.8561  0.8790 0.0092  [0.855 0.861 0.869 0.875 0.877]            0.8767  0.8734  0.8801 0.0027  [0.873 0.875 0.877 0.878 0.881]
lgb_embeds: mles_hyper20_hyperbolic_060.pickle           0.8685  0.8570  0.8800 0.0093  [0.854 0.868 0.869 0.875 0.877]            0.8779  0.8728  0.8829 0.0041  [0.874 0.875 0.877 0.879 0.884]
lgb_embeds: mles_hyper20_hyperbolic_070.pickle           0.8728  0.8622  0.8833 0.0085  [0.863 0.867 0.871 0.881 0.882]            0.8796  0.8782  0.8811 0.0012  [0.879 0.879 0.880 0.880 0.881]
lgb_embeds: mles_hyper20_hyperbolic_080.pickle           0.8710  0.8594  0.8826 0.0093  [0.858 0.869 0.869 0.877 0.882]            0.8796  0.8764  0.8827 0.0025  [0.876 0.877 0.881 0.882 0.882]
lgb_embeds: mles_hyper20_hyperbolic_090.pickle           0.8721  0.8608  0.8833 0.0091  [0.861 0.868 0.869 0.879 0.884]            0.8765  0.8729  0.8802 0.0029  [0.874 0.875 0.875 0.879 0.880]
lgb_embeds: mles_hyper20_hyperbolic_100.pickle           0.8707  0.8611  0.8804 0.0078  [0.859 0.869 0.871 0.877 0.878]            0.8769  0.8739  0.8799 0.0024  [0.875 0.875 0.876 0.877 0.881]
lgb_embeds: mles_hyper20_hyperbolic_110.pickle           0.8702  0.8604  0.8799 0.0079  [0.859 0.868 0.868 0.877 0.878]            0.8770  0.8739  0.8801 0.0025  [0.874 0.876 0.876 0.878 0.881]
lgb_embeds: mles_hyper20_hyperbolic_120.pickle           0.8700  0.8604  0.8796 0.0077  [0.859 0.865 0.872 0.874 0.879]            0.8795  0.8762  0.8829 0.0027  [0.877 0.878 0.879 0.882 0.883]
lgb_embeds: mles_hyper20_hyperbolic_130.pickle           0.8725  0.8642  0.8808 0.0067  [0.865 0.869 0.869 0.879 0.880]            0.8777  0.8751  0.8803 0.0021  [0.875 0.877 0.878 0.879 0.880]
lgb_embeds: mles_hyper20_hyperbolic_140.pickle           0.8715  0.8608  0.8821 0.0085  [0.859 0.868 0.872 0.878 0.880]            0.8815  0.8769  0.8860 0.0036  [0.879 0.879 0.880 0.881 0.888]
lgb_embeds: mles_hyper20_hyperbolic_150.pickle           0.8722  0.8636  0.8808 0.0069  [0.863 0.868 0.873 0.879 0.879]            0.8756  0.8740  0.8772 0.0013  [0.874 0.874 0.876 0.876 0.877]

lgb_embeds: mles_hyper12_poincare_010.pickle           0.8549  0.8466  0.8633 0.0067  [0.848 0.848 0.856 0.859 0.863]            0.8712  0.8660  0.8764 0.0042  [0.867 0.869 0.871 0.871 0.878]
lgb_embeds: mles_hyper12_poincare_020.pickle           0.8619  0.8532  0.8705 0.0070  [0.853 0.857 0.862 0.868 0.869]            0.8720  0.8674  0.8765 0.0037  [0.868 0.870 0.871 0.875 0.877]
lgb_embeds: mles_hyper12_poincare_030.pickle           0.8652  0.8552  0.8753 0.0081  [0.857 0.860 0.861 0.874 0.874]            0.8758  0.8712  0.8803 0.0036  [0.871 0.874 0.875 0.878 0.880]
lgb_embeds: mles_hyper12_poincare_040.pickle           0.8627  0.8505  0.8749 0.0098  [0.852 0.854 0.863 0.871 0.874]            0.8726  0.8678  0.8775 0.0039  [0.869 0.869 0.871 0.876 0.878]
lgb_embeds: mles_hyper12_poincare_050.pickle           0.8691  0.8558  0.8824 0.0107  [0.856 0.861 0.869 0.878 0.881]            0.8810  0.8767  0.8852 0.0034  [0.877 0.879 0.880 0.883 0.886]
lgb_embeds: mles_hyper12_poincare_060.pickle           0.8690  0.8568  0.8812 0.0098  [0.856 0.864 0.867 0.879 0.879]            0.8804  0.8774  0.8834 0.0024  [0.879 0.879 0.879 0.881 0.884]
lgb_embeds: mles_hyper12_poincare_070.pickle           0.8676  0.8544  0.8809 0.0107  [0.855 0.862 0.864 0.878 0.879]            0.8782  0.8755  0.8809 0.0022  [0.875 0.878 0.878 0.879 0.881]
lgb_embeds: mles_hyper12_poincare_080.pickle           0.8669  0.8543  0.8794 0.0101  [0.856 0.858 0.865 0.877 0.878]            0.8750  0.8699  0.8800 0.0041  [0.871 0.873 0.873 0.877 0.881]
lgb_embeds: mles_hyper12_poincare_090.pickle           0.8657  0.8529  0.8785 0.0103  [0.856 0.856 0.863 0.876 0.877]            0.8781  0.8734  0.8828 0.0038  [0.874 0.875 0.878 0.881 0.883]
lgb_embeds: mles_hyper12_poincare_100.pickle           0.8680  0.8556  0.8804 0.0100  [0.855 0.861 0.869 0.877 0.878]            0.8788  0.8760  0.8816 0.0023  [0.876 0.878 0.878 0.880 0.882]
lgb_embeds: mles_hyper12_poincare_110.pickle           0.8692  0.8574  0.8810 0.0095  [0.858 0.862 0.869 0.878 0.879]            0.8802  0.8753  0.8851 0.0040  [0.876 0.877 0.880 0.883 0.885]
lgb_embeds: mles_hyper12_poincare_120.pickle           0.8693  0.8574  0.8812 0.0096  [0.859 0.860 0.870 0.877 0.880]            0.8754  0.8727  0.8782 0.0022  [0.873 0.874 0.875 0.876 0.879]
lgb_embeds: mles_hyper12_poincare_130.pickle           0.8664  0.8547  0.8780 0.0094  [0.853 0.860 0.868 0.875 0.875]            0.8773  0.8731  0.8815 0.0034  [0.874 0.875 0.877 0.879 0.882]
lgb_embeds: mles_hyper12_poincare_140.pickle           0.8695  0.8577  0.8812 0.0095  [0.861 0.861 0.867 0.878 0.881]            0.8837  0.8814  0.8861 0.0019  [0.882 0.883 0.883 0.884 0.887]
lgb_embeds: mles_hyper12_poincare_150.pickle           0.8671  0.8564  0.8777 0.0086  [0.859 0.859 0.865 0.876 0.876]            0.8756  0.8704  0.8809 0.0042  [0.870 0.873 0.878 0.878 0.880]

lgb_embeds: mles_hyper22_poincare_010.pickle           0.7919  0.7804  0.8034 0.0093  [0.777 0.791 0.793 0.799 0.800]            0.7917  0.7850  0.7984 0.0054  [0.785 0.788 0.791 0.795 0.799]
lgb_embeds: mles_hyper22_poincare_020.pickle           0.7959  0.7845  0.8073 0.0092  [0.781 0.794 0.799 0.802 0.803]            0.8082  0.8019  0.8145 0.0051  [0.803 0.804 0.809 0.810 0.816]
lgb_embeds: mles_hyper22_poincare_030.pickle           0.8023  0.7901  0.8146 0.0098  [0.787 0.799 0.806 0.808 0.811]            0.8137  0.8110  0.8164 0.0022  [0.812 0.812 0.813 0.815 0.817]
lgb_embeds: mles_hyper22_poincare_040.pickle           0.7972  0.7875  0.8069 0.0078  [0.783 0.799 0.800 0.801 0.803]            0.8102  0.8064  0.8141 0.0031  [0.808 0.808 0.809 0.811 0.815]
lgb_embeds: mles_hyper22_poincare_050.pickle           0.7980  0.7853  0.8106 0.0102  [0.781 0.798 0.799 0.804 0.807]            0.8132  0.8070  0.8193 0.0049  [0.807 0.812 0.813 0.813 0.821]
lgb_embeds: mles_hyper22_poincare_060.pickle           0.8029  0.7923  0.8134 0.0085  [0.788 0.804 0.805 0.808 0.809]            0.8161  0.8116  0.8207 0.0037  [0.812 0.813 0.817 0.817 0.821]
lgb_embeds: mles_hyper22_poincare_070.pickle           0.7963  0.7866  0.8061 0.0079  [0.783 0.798 0.799 0.800 0.802]            0.8182  0.8148  0.8215 0.0027  [0.815 0.816 0.818 0.820 0.822]
lgb_embeds: mles_hyper22_poincare_080.pickle           0.7992  0.7873  0.8111 0.0096  [0.782 0.800 0.803 0.805 0.806]            0.8203  0.8138  0.8268 0.0052  [0.813 0.816 0.823 0.823 0.826]
lgb_embeds: mles_hyper22_poincare_090.pickle           0.7978  0.7869  0.8087 0.0088  [0.783 0.799 0.800 0.803 0.804]            0.8198  0.8151  0.8246 0.0038  [0.816 0.818 0.818 0.821 0.826]
lgb_embeds: mles_hyper22_poincare_100.pickle           0.7952  0.7881  0.8024 0.0058  [0.785 0.797 0.797 0.798 0.800]            0.8173  0.8138  0.8208 0.0028  [0.813 0.815 0.818 0.820 0.820]
lgb_embeds: mles_hyper22_poincare_110.pickle           0.7979  0.7843  0.8116 0.0110  [0.779 0.801 0.802 0.802 0.806]            0.8214  0.8169  0.8259 0.0036  [0.817 0.819 0.822 0.823 0.826]
lgb_embeds: mles_hyper22_poincare_120.pickle           0.7957  0.7866  0.8048 0.0073  [0.784 0.795 0.797 0.801 0.802]            0.8171  0.8122  0.8221 0.0040  [0.812 0.815 0.817 0.818 0.823]
lgb_embeds: mles_hyper22_poincare_130.pickle           0.7990  0.7880  0.8100 0.0089  [0.784 0.801 0.801 0.803 0.806]            0.8235  0.8186  0.8285 0.0040  [0.819 0.822 0.823 0.824 0.830]
lgb_embeds: mles_hyper22_poincare_140.pickle           0.7978  0.7881  0.8075 0.0078  [0.784 0.800 0.800 0.803 0.803]            0.8134  0.8102  0.8166 0.0026  [0.810 0.812 0.814 0.815 0.817]
lgb_embeds: mles_hyper22_poincare_150.pickle           0.7927  0.7810  0.8044 0.0094  [0.777 0.793 0.796 0.796 0.801]            0.8134  0.8081  0.8187 0.0043  [0.806 0.814 0.814 0.815 0.818]

lgb_embeds: mles_hyper24_poincare_010.pickle           0.8610  0.8506  0.8714 0.0084  [0.852 0.853 0.862 0.868 0.870]            0.8659  0.8618  0.8701 0.0034  [0.862 0.866 0.866 0.866 0.871]
lgb_embeds: mles_hyper24_poincare_020.pickle           0.8644  0.8523  0.8765 0.0097  [0.853 0.856 0.866 0.873 0.874]            0.8701  0.8653  0.8749 0.0039  [0.867 0.869 0.869 0.870 0.877]
lgb_embeds: mles_hyper24_poincare_030.pickle           0.8675  0.8562  0.8787 0.0091  [0.856 0.861 0.868 0.875 0.877]            0.8778  0.8715  0.8841 0.0051  [0.872 0.874 0.879 0.879 0.885]
lgb_embeds: mles_hyper24_poincare_040.pickle           0.8722  0.8632  0.8812 0.0073  [0.864 0.868 0.870 0.879 0.881]            0.8764  0.8717  0.8811 0.0038  [0.872 0.875 0.876 0.877 0.882]
lgb_embeds: mles_hyper24_poincare_050.pickle           0.8699  0.8610  0.8789 0.0072  [0.861 0.864 0.871 0.876 0.878]            0.8771  0.8733  0.8808 0.0030  [0.874 0.875 0.877 0.877 0.882]
lgb_embeds: mles_hyper24_poincare_060.pickle           0.8710  0.8613  0.8807 0.0078  [0.863 0.863 0.872 0.877 0.880]            0.8805  0.8756  0.8854 0.0039  [0.875 0.879 0.881 0.882 0.886]
lgb_embeds: mles_hyper24_poincare_070.pickle           0.8713  0.8633  0.8793 0.0064  [0.865 0.867 0.868 0.878 0.879]            0.8812  0.8774  0.8850 0.0031  [0.879 0.880 0.880 0.882 0.886]
lgb_embeds: mles_hyper24_poincare_080.pickle           0.8690  0.8584  0.8795 0.0085  [0.860 0.861 0.869 0.876 0.879]            0.8804  0.8766  0.8842 0.0031  [0.878 0.879 0.879 0.882 0.885]
lgb_embeds: mles_hyper24_poincare_090.pickle           0.8713  0.8609  0.8818 0.0084  [0.861 0.867 0.869 0.879 0.880]            0.8743  0.8712  0.8775 0.0025  [0.870 0.875 0.875 0.876 0.876]
lgb_embeds: mles_hyper24_poincare_100.pickle           0.8720  0.8627  0.8813 0.0075  [0.862 0.868 0.873 0.876 0.882]            0.8811  0.8782  0.8840 0.0024  [0.878 0.879 0.882 0.882 0.884]
lgb_embeds: mles_hyper24_poincare_110.pickle           0.8723  0.8624  0.8822 0.0080  [0.862 0.867 0.873 0.879 0.880]            0.8807  0.8789  0.8826 0.0015  [0.879 0.880 0.880 0.882 0.882]
lgb_embeds: mles_hyper24_poincare_120.pickle           0.8730  0.8632  0.8829 0.0079  [0.864 0.867 0.874 0.876 0.884]            0.8797  0.8755  0.8840 0.0034  [0.876 0.877 0.879 0.882 0.884]
lgb_embeds: mles_hyper24_poincare_130.pickle           0.8730  0.8614  0.8845 0.0093  [0.862 0.864 0.874 0.881 0.883]            0.8783  0.8740  0.8827 0.0035  [0.873 0.878 0.879 0.881 0.881]
lgb_embeds: mles_hyper24_poincare_140.pickle           0.8733  0.8620  0.8847 0.0091  [0.864 0.866 0.871 0.882 0.883]            0.8816  0.8771  0.8861 0.0036  [0.878 0.880 0.880 0.882 0.888]
lgb_embeds: mles_hyper24_poincare_150.pickle           0.8710  0.8591  0.8828 0.0095  [0.858 0.865 0.872 0.879 0.880]            0.8779  0.8749  0.8809 0.0024  [0.875 0.877 0.878 0.878 0.882]
