LOGDIR := "/content/drive/MyDrive/colab_training/logs/detr"

train_cpu: FORCE
	python3 train.py \
		--device=cpu \
		--epochs=10 \
		--logdir=$(LOGDIR) \
		--train_batch=4 \
		--val_batch=16 \
		--val_interval=50 \
		--download
FORCE:

train_gpu: FORCE
	python3 train.py \
		--device=gpu \
		--epochs=100 \
		--logdir=$(LOGDIR) \
		--train_batch=128 \
		--val_batch=32 \
		--val_interval=20 \
		--download
FORCE:
