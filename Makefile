dataset:
	mkdir -p ./data/processed
	python ./src/prepare_dataset.py	$(n_images)
train:
	mkdir -p models
	mkdir -p results
	python ./src/02_train_model.py >> ./results/dtree_output.txt

train_rf:
	mkdir -p models
	mkdir -p results
	python ./src/02c_train_rf.py >> ./results/rf_results/rf_output.txt

predict:
	python ./src/03_generate_results.py $(model) $(img)

scores:
	python ./src/04_segmentation_accuracy.py $(pred_img) $(real_img)

multi_otsu:
	python ./src/multi_otsu.py $(img)
