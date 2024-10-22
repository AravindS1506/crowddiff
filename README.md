# Crowd-Density-Estimation-EE798R
[Crowddiff Paper CVPR](https://arxiv.org/pdf/2303.12790)
         [Official Implementation](https://github.com/dylran/crowddiff.git)
         
**Report.pdf** contains the information about implementation, datasets and results.
## Instructions to implement:
### Pre-Process
1) Clone the repository
   
```bash
git clone https://github.com/AravindS1506/crowddiff.git
```
2) Create a conda environment and install the requirements.txt file
```bash
cd crowddiff
pip install -r requirements.txt
apt-get update && apt-get install -y libopenmpi-dev openmpi-bin
pip install mpi4py
```

3) Make the model and datasets directory and pre-processed dataset library
```bash
mkdir dataset
mkdir model
mkdir out_data
```

4) Download the [pre-trained checkpoints](https://drive.google.com/file/d/1dLEjaZqw9bxQm2sUU4I6YXDnFfyEHl8p/view?usp=sharing) and place it in the model folder created 

5) Download the [datasets](https://drive.google.com/drive/folders/1D4Bs4YuKztg9iPvPnEhZNLRYPjgY3bTK?usp=sharing) and place them in the dataset folder.
   
6) Run the preprocessing script for each dataset using the following script:
```bash
python cc_utils/preprocess_jhu.py --data_dir  dataset --output_dir out_data --dataset jhu --mode test --image_size 256 --ndevices 1 --sigma '0.5'  --kernel_size '3'
```
Replace the dataset name as required in the above code. Ensure that the dataset is organized in the format dataset/dataset_name and use --data_dir as dataset and --dataset as dataset_name. "preprocess_jhu" is used for jhu_crowd++, "preprocess_ucf" is used for UCF_CC_50 and ucf_qnrf, "preprocess_shtech" is used for shtech_A and shtech_B.



7) Once preprocessing is done, place "cc_utils" in the folder "scripts". This is required for the training and testing part of the program.
### Training
8) Training is not done in this **implementation** due to the availability of pre-trained weights and its computation complexity. However, training can be done by using the following code:
```bash
DATA_DIR="--data_dir path/to/train/data --val_samples_dir path/to/val/data"
LOG_DIR="--log_dir path/to/results --resume_checkpoint path/to/pre-trained/weights"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 8 --save_interval 10000 --lr 1e-4"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"

CUDA_VISIBLE_DEVICES=0 python scripts/super_res_train.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```
### Testing
9) For testing purposes, run the code; the code below is for the jhu_crowd++ dataset. Replace the path as required. This will output the MAE and MSE after each image has been processed.
```bash
DATA_DIR="--data_dir out_data/jhu/part_1/test/"
LOG_DIR="--log_dir results --model_path model/demo.pt"
TRAIN_FLAGS="--normalizer 0.8 --pred_channels 1 --batch_size 1 --per_samples 1"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --large_size 256  --small_size 256 --learn_sigma True --noise_schedule linear --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_checkpoint True"
CUDA_VISIBLE_DEVICES=0 python scripts/super_res_sample.py $DATA_DIR $LOG_DIR $TRAIN_FLAGS $MODEL_FLAGS
```

### Simulations
To obtain a combined image with the pre-processed image and a density map, process an image using the pre-process script. Choose a single image among the ones generated from the script and run the testing script. This will save a pred_density map. Run the following code to combine the actual image and the predicted density map. Change the paths in the program accordingly
```bash
python merge.py
```
