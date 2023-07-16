LOGDIR := "/content/drive/MyDrive/colab_training/logs/detr"
IMAGE := "d495374ddbd1"
HOST_DIR := "/tmp/detr"
CONT_DIR := "/home/temp"
PORT := 55502
CUDA_VISIBLE_DEVICES := "4"

train: FORCE
	python3 train.py \
		--device=cpu \
		--epochs=10 \
		--logdir=$(CONT_DIR) \
		--train_batch=4 \
		--val_batch=16 \
		--val_interval=50 \
		--download
FORCE:

train_cloud: FORCE
	python3 train.py \
		--device=gpu \
		--epochs=100 \
		--logdir=$(LOGDIR) \
		--train_batch=128 \
		--val_batch=32 \
		--val_interval=20 \
		--download
FORCE:

train_container: FORCE
	python3 train.py \
		--device=gpu \
		--epochs=40 \
		--logdir=$(CONT_DIR) \
		--train_batch=64 \
		--val_batch=32 \
		--val_interval=40 \
		--download
FORCE:

image: FORCE
	docker build -t detr_training docker
FORCE:

push_image: FORCE
	docker login
	docker tag $(IMAGE) bitofplastic/images:detr_train_v0
	docker push bitofplastic/images:detr_train_v0
FORCE:

pull_image: FORCE
	docker pull bitofplastic/images:detr_train_v0
FORCE:

container: FORCE
	mkdir -p $(HOST_DIR)
	docker run \
		-it \
		-d \
		--name detr_lab \
		-p $(PORT):$(PORT) \
		-v $(HOST_DIR):$(CONT_DIR) \
		--env CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
		bitofplastic/images:detr_train_v0
FORCE:

attach: FORCE
	docker start -ia detr_lab
FORCE:
