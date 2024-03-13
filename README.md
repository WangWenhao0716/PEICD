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
