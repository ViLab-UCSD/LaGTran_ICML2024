dataset=$1
source=$2
target=$3

# values=("0.1" "0.2" "0.3" "0.5" "0.75" "0.9")
values=("1.0")

for val in "${values[@]}"; do
    python3 text_classification_bert.py --dataset $1 --source $2 --target $3 --n_iters 20000 --batch_size 24 --root_dir /newdata/tarun/datasets/GeoNet/metadata/${1} --sample --frac ${val} --learning 3e-3
done