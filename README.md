# PEICD
The official implementation of "Pattern-Expandable Image Copy Detection"

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

5. Download the novel query set

6. Download the reference set



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
