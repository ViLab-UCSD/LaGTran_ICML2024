# dataset=$1
# source=$2
# target=$3
# n_iters=$4
# batch_size=$5
# adapt=$6
# lr=$7
# root="/longtail-ssl/Datasets/DomainNet/"

# cd /longtail-vol/tarun/LangBasedGeoDA/

# values=(0.1 0.2 0.3 0.5 0.75 0.9)

# source="real"
# for val in "${values[@]}"; do
#     target="clipart"
#     python3 text_classification_dnet.py --dataset ${dataset} --source ${source} --target ${target} --n_iters ${n_iters} --batch_size ${batch_size} --learning_rate ${lr} --root_dir ${root} --sample --frac ${val}
    
#     target="sketch"
#     python3 text_classification_dnet.py --dataset ${dataset} --source ${source} --target ${target} --n_iters 0 --batch_size ${batch_size} --learning_rate ${lr} --root_dir ${root} --sample --frac ${val}
    
#     target="painting"
#     python3 text_classification_dnet.py --dataset ${dataset} --source ${source} --target ${target} --n_iters 0 --batch_size ${batch_size} --learning_rate ${lr} --root_dir ${root} --sample --frac ${val}
    
# done

python3 text_classification_dnet.py --dataset officeHome --source real --target art --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source real --target product --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source real --target clipart --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&

python3 text_classification_dnet.py --dataset officeHome --source art --target real --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source art --target product --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source art --target clipart --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&

python3 text_classification_dnet.py --dataset officeHome --source product --target real --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source product --target art --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source product --target clipart --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&

python3 text_classification_dnet.py --dataset officeHome --source clipart --target real --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source clipart --target art --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample &&
python3 text_classification_dnet.py --dataset officeHome --source clipart --target product --n_iters 0000 --batch_size 24 --learning_rate 5e-5 --root_dir /newfoundland/tarun/datasets/Adaptation/OfficeHome/Dataset10072016 --sample