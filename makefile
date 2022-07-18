base_url = "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0"

test:
	python -m src.test_run
predict:
	python -m src.predict
train:
	python -m src.train
download_train_set:
	mkdir -p sets/train
	cd sets/train; \
		wget 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip'; \
		unzip DIV2K_train_HR.zip
download_model:
	mkdir -p _models
	cd _models; \
		wget $(base_url)/fbcnn_color.pth; \
		wget $(base_url)/fbcnn_gray.pth; \
		wget $(base_url)/fbcnn_gray_double.pth
