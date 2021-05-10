dataset:
	mkdir -p ./data/processed
	python ./src/prepare_dataset.py	$(n_images)
train:
	mkdir -p models
	mkdir -p results
	python ./src/02_train_model.py >> ./results/dtree_output.txt
