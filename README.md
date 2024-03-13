# PEICD
[In Submission] The official implementation of "Pattern-Expandable Image Copy Detection"

![image](https://github.com/WangWenhao0716/PEICD/blob/main/PEICD.png)

# Prepare Datasets

1. Download the base training set
  ```
for letter in {a..c}; do
    wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_name_part_a$letter
done
cat train_v1_name_part_aa train_v1_name_part_ab train_v1_name_part_ac > train_v1_name.tar


  ```

3. Download the novel training set
  ```
for letter in {a..c}; do
    wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/train_v1_name_same_part_a$letter
done
cat train_v1_name_same_part_aa train_v1_name_same_part_ab train_v1_name_same_part_ac > train_v1_name_same.tar
  ```
4. Download the base query set
```
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/query_v1_pattern_same_clean.tar
```
6. Download the novel query set
```
wget https://huggingface.co/datasets/WenhaoWang/PE-ICD/resolve/main/query_v1_pattern_clean.tar
```
7. Download the reference set

```
for i in {0..19}
do
   wget https://dl.fbaipublicfiles.com/image_similarity_challenge/public/references_$i.zip
done
```

# Prepare Environment
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


# Train
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
!CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_tune.py \
-ds train_v1_name_same -a vit_base_pattern_tune --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /path/to/data/ \
--logs-dir logs/train_v1_name/vit_base_pattern_prompt_minus_4_tune/ \
--begin logs/train_v1_name/vit_base_pattern_prompt_minus_4/checkpoint_24.pth.tar \
--height 224 --width 224
```



# Citation
```
@inproceedings{
    wang2024peicd,
    title={Pattern-Expandable Image Copy Detection},
    author={Wang Wenhao and Sun Yifan and Yang Yi},
    booktitle={In submission},
    year={2024},
}
```
