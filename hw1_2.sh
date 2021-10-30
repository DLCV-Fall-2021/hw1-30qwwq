wget -O 'hw1_2_model.pth' https://www.dropbox.com/s/ptihbvicwn59axd/model_hw1_2.pth?dl=1
python3 problem2/test.py --img_dir $1 --save_dir $2 --model_path hw1_2_model.pth