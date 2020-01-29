# Setup

```sh
# download Miniconda from https://conda.io/
curl -OL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
# execute intall file
bash Miniconda*.sh
# relogin to apply PATH changes
exit

# install pytorch
conda install pytorch -c pytorch
# check CUDA version
python -c 'import torch; print(torch.version.cuda)'
# chech torch GPU
# See question: https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu
python -c 'import torch; print(torch.rand(2,3).cuda())'

# clone repo
git clone git@bitbucket.org:dllllb/dltranz.git

# install dependencies
cd dltranz
pip install -r requirements.txt

# download datasets
./get-data.sh

# convert datasets from transaction list to features for metric learning
./make-datasets.sh
```

# Run scenario

## age-pred dataset

```sh
# Train metric learning model
dltrans/opends$ python metric_learning.py --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_train.json

# With pretrained mertic learning model run inference ang take embeddings for each customer
dltrans/opends$ python ml_inference.py --conf conf/age_pred_ml_dataset.hocon conf/age_pred_ml_params_inference.json

# Train supervised model and save scores to file
dltrans/opends$ python -m scenario_age_pred fit_target --conf conf/age_pred_target_dataset.hocon conf/age_pred_target_params_train.json

# Take pretrained ml model and fine tune it in supervised mode and save scores to file
dltrans/opends$ python -m scenario_age_pred fit_finetuning --conf conf/age_pred_target_dataset.hocon conf/age_pred_finetuning_params_train.json

# Run estimation for different approaches
# Check some options with `--help` argument
dltrans/opends$ python -m scenario_age_pred compare_approaches

# check the results
dltrans/opends$ cat runs/scenario_age_pred.csv
```

## gender dataset

```sh
cd dltrans/opends

# Train metric learning model
python metric_learning.py --conf conf/gender_dataset.hocon conf/gender_ml_params_train.json

# With pretrained mertic learning model run inference ang take embeddings for each customer
python ml_inference.py --conf conf/gender_dataset.hocon conf/gender_ml_params_inference.json

# Train supervised model and save scores to file
python -m scenario_gender fit_target --conf conf/gender_dataset.hocon conf/gender_target_params_train.json

# Take pretrained ml model and fine tune it in supervised mode and save scores to file
python -m scenario_gender fit_finetuning --conf conf/gender_dataset.hocon conf/gender_finetuning_params_train.json

# Run estimation for different approaches
# Check some options with `--help` argument
python -m scenario_gender compare_approaches

# check the results
cat runs/scenario_gender.csv
```

## tinkoff dataset

```sh
cd dltrans/opends

# Train metric learning model
python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json

# With pretrained mertic learning model run inference ang take embeddings for each customer
python ml_inference.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_inference_params.json

# Run estimation for different approaches
# Check some options with `--help` argument
dltrans/opends $ 
rm runs/scenario_tinkoff.json

python -m scenario_tinkoff train --name 'baseline_const' --user_layers 1 --item_layers 1 --max_epoch 2

python -m scenario_tinkoff train --name 'no user features' --user_layers 1 --item_layers E
python -m scenario_tinkoff train --name 'ml embeddings'  --use_embeddings --user_layers 1T --item_layers E1 --optim_weight_decay 0.0001  0.1 0.1  0.00001 --optim_lr 0.005
python -m scenario_tinkoff train --name 'transactional stat'  --use_trans_common_features --use_trans_mcc_features --user_layers 1T --item_layers E1 --optim_weight_decay 0.0001  1 1  0.00001 --optim_lr 0.005
python -m scenario_tinkoff train --name 'social demograpy' --use_gender --user_layers 1T --item_layers E1 --optim_weight_decay 0.0001  1 1  0.00001 --optim_lr 0.005

# check the results
python -m scenario_tinkoff convert_history_file --report_file "runs/scenario_tinkoff.csv"

cat "runs/scenario_tinkoff.csv"
```

## common scenario (work in progress)

### Train metric learning model

Run this script for every project: age-pred, tinkoff, gender.
Provide valid dataset description in hocon file and model params in json file

```sh
python metric_learning.py --conf conf/tinkoff_dataset.hocon conf/tinkoff_train_params.json
```

### With pretrained mertic learning model run inference ang take embeddings for each customer

Run this script for every project: age-pred, tinkoff, gender.
Provide valid dataset description in hocon file and model inference params in json file

```sh
python ml_inference.py --conf conf/dataset.hocon conf/ml_params_inference.json
```

### Use embeddings for each customer as a features for specific machine learning problem

See code in notebooks in `notebooks/`.
