1. Use Symbolic learning model initially 
python symbolic_model_train_768_semantic_embeddings_v2.py train --ckpt_out symbolic_binary_prompt.pt --fp16 --batch_size 4096 --epochs 2000 --lr 3e-4 --use_offline_pseudo_aug --use_pseudo_negative_aug --plot_curves 

2. Use VLM to filter 
Use VLM to produce predictions (in VLM_predictions) on whether there are anomalies in video segments. Refer to them, especially negative decisions, to filter out false positives from the symbolic learning model 

