# PEICD
[In Submission] The official implementation of "Pattern-Expandable Image Copy Detection"

![image](https://github.com/WangWenhao0716/PEICD/blob/main/PEICD.png)

## Prepare Datasets

1. Download the base training set
  ```
for letter in {a..c}; do
    wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_name_part_a$letter
done
cat train_v1_name_part_aa train_v1_name_part_ab train_v1_name_part_ac > train_v1_name.tar

wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_labels_num.npy
  ```

3. Download the novel training set
  ```
for letter in {a..c}; do
    wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_name_same_part_a$letter
done
cat train_v1_name_same_part_aa train_v1_name_same_part_ab train_v1_name_same_part_ac > train_v1_name_same.tar

wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_labels_num_same.npy
  ```
4. Download the base query set
```
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/query_v1_pattern_same_clean.tar
```
5. Download the novel query set
```
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/query_v1_pattern_clean.tar
```
6. Download the reference set

```
for i in {0..19}
do
   wget https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_$i.zip
done
```
7. Download the original images

```
for i in {0..19}
do
   wget https://dl.fbaipublicfiles.com/image_similarity_challenge/public/train_$i.zip
done
```

Note: you can also generate the base and novel training images using the original images with ```train_v1_name_same.py``` and ```train_v1_name.py```.

## Prepare Environment
You can directly download our curated environment by
```
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/torch21.tar
```
then
```
tar -xvf torch21.tar
export PATH="$(pwd)/torch21/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/torch21/lib:$LD_LIBRARY_PATH"
```

Or, you can prepare an environment by yourself: our method only relies on basic libraries, such as PyTorch.


## Train
The training code is available at ```train```. 
You can perform two stages of training by
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema.py \
-ds train_v1_name -a vit_base_pattern --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /path/to/data/ \
--logs-dir logs/train_v1_name/vit_base_pattern_prompt_minus_4 \
--height 224 --width 224
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_tune.py \
-ds train_v1_name_same -a vit_base_pattern_tune --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /path/to/data/ \
--logs-dir logs/train_v1_name/vit_base_pattern_prompt_minus_4_tune/ \
--begin logs/train_v1_name/vit_base_pattern_prompt_minus_4/checkpoint_24.pth.tar \
--height 224 --width 224
```

## Test
We denote the two trained models as ```train_v1_vit_base_pattern_4_minus.pth.tar``` and ```train_v1_vit_base_pattern_tune_4_minus.pth.tar```, or you can directly download it by

```
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_vit_base_pattern_4_minus.pth.tar
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_vit_base_pattern_tune_4_minus.pth.tar
```

The test code is available at ```test```. 

Extract features of queries generated by base patterns:
```
mkdir -p ./feature/train_v1_vit_base_pattern_tune_4_minus
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/query_v1_pattern_same_clean \
      --o ./feature/train_v1_vit_base_pattern_tune_4_minus/query_v1_same.hdf5 \
      --model vit_base_pattern_tune  --GeM_p 3 \
      --checkpoint train_v1_vit_base_pattern_tune_4_minus.pth.tar --imsize 224 
```
Extract features of queries generated by novel patterns:
```
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/query_v1_pattern_clean \
      --o ./feature/train_v1_vit_base_pattern_tune_4_minus/query_v1.hdf5 \
      --model vit_base_pattern_tune  --GeM_p 3 \
      --checkpoint train_v1_vit_base_pattern_tune_4_minus.pth.tar --imsize 224 
```
Extract out-of-date features of references:
```
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/reference_images \
      --o ./feature/train_v1_vit_base_pattern_tune_4_minus/reference_v1_cold.hdf5 \
      --model vit_base_pattern  --GeM_p 3 \
      --checkpoint train_v1_vit_base_pattern_4_minus.pth.tar --imsize 224 
```

Extract out-of-date features of original images:
```
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/training_images \
      --o ./feature/train_v1_vit_base_pattern_tune_4_minus/training_v1_cold.hdf5 \
      --model vit_base_pattern  --GeM_p 3 \
      --checkpoint train_v1_vit_base_pattern_4_minus.pth.tar --imsize 224 
```

Matching with base patterns
```
python score_normalization.py \
    --query_descs ./feature/train_v1_vit_base_pattern_tune_4_minus/query_{0..4}_v1_same.hdf5\
    --db_descs ./feature/train_v1_vit_base_pattern_tune_4_minus/reference_{0..99}_v1_cold.hdf5 \
    --train_descs ./feature/train_v1_vit_base_pattern_tune_4_minus/training_{0..99}_v1_cold.hdf5 \
    --factor 2 --n 10 \
    --o ./feature/train_v1_vit_base_pattern_tune_4_minus/predictions_train_v1_same_cold.csv \
    --reduction avg --max_results 500_000

python compute_metrics.py \
--preds_filepath ./feature/train_v1_vit_base_pattern_tune_4_minus/predictions_train_v1_same_cold.csv \
--gt_filepath ./gt_v1.csv
```
This should give
```
Track 1 results of 499999 predictions (10000 GT matches)
Average Precision: 0.90683
Recall at P90    : 0.87990
Threshold at P90 : -0.110044
Recall at rank 1:  0.94170
Recall at rank 10: 0.95430
```

Matching with novel patterns

```
python score_normalization.py \
    --query_descs ./feature/train_v1_vit_base_pattern_tune_4_minus/query_{0..4}_v1.hdf5\
    --db_descs ./feature/train_v1_vit_base_pattern_tune_4_minus/reference_{0..99}_v1_cold.hdf5 \
    --train_descs ./feature/train_v1_vit_base_pattern_tune_4_minus/training_{0..99}_v1_cold.hdf5 \
    --factor 2 --n 10 \
    --o ./feature/train_v1_vit_base_pattern_tune_4_minus/predictions_train_v1_cold.csv \
    --reduction avg --max_results 500_000

python compute_metrics.py \
--preds_filepath ./feature/train_v1_vit_base_pattern_tune_4_minus/predictions_train_v1_cold.csv \
--gt_filepath ./gt_v1.csv
```
This should give
```
Track 1 results of 500000 predictions (10000 GT matches)
Average Precision: 0.73324
Recall at P90    : 0.65560
Threshold at P90 : -0.111441
Recall at rank 1:  0.81130
Recall at rank 10: 0.85830
```


## Citation
```
@inproceedings{
    wang2024peicd,
    title={Pattern-Expandable Image Copy Detection},
    author={Wang, Wenhao and Sun, Yifan and Yang, Yi},
    booktitle={In submission},
    year={2024},
}
```

## Contact

If you have any questions, feel free to contact Wenhao Wang (wangwenhao0716@gmail.com).
