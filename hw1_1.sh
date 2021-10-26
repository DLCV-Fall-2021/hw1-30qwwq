wget -O 'hw1_1_model.pth' https://www.dropbox.com/s/jp0jw9lkqzoqoc6/model_hw1_1.pth?dl=1  
python3 problem1/test.py --img_dir $1 --save_dir $2 --model_path hw1_1_model.pth