# FL-DiG

This is the implement of FL-DiG. We trained with MT and NEU-CLS dataset.

## Install Requirement

opencv>==6.5.5
torch>=1.7.0
python>=3.6
scikit-fuzzy>=0.4.2
scikit-learn>=0.24.2

## How To Run


#### Step 1: Training DCDM

python3 DCDM/DCDM_Main.py --state train --device cuda:0 --epochs 200 --train_datadir /path/to/train --save_F_datadir data/NEU-CLS/F_train --save_weight_dir test2/model

#### Step 2: Generation Data using DCDM

python3 DCDM/DCDM_Main.py --state test --device cuda:0 --save_F_datadir data/NEU-CLS/F_train --save_weight_dir test2/model --g_data_dir data/NEU-CLS/Gdata

#### Step 3: Training FL

python3 main.py --device cuda:0 --net resnet50 --class_num 6 --client_num 2 --client_class_num 3  --num_epochs 50 --align_datadir data/NEU-CLS/Gdata
