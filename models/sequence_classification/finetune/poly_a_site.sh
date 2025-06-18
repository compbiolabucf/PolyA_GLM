data_path="/home/sourav/Poly_A_Site/data/Human/intergene"
lr=1e-6
MAX_LENGTH=26
seed=42

echo "The provided data_path is $data_path"
python train.py \
            --model_name_or_path zhihan1996/DNA_bert_6\
            --data_path  ${data_path}/$data \
            --kmer 6 \
            --run_name DNABERT2_${data}_seed${seed} \
            --model_max_length 512 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --learning_rate 1e-6 \
            --num_train_epochs 3 \
            --fp16 \
            --save_steps 200 \
            --output_dir output_intergene-gene_modified/DNABERT3_101_epoch3_early_gene_withwarmupcyclic_updatedfold_changed_version\
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 100000 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            # --find_unused_parameters False