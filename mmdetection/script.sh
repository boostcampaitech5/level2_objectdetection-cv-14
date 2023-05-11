python train_base.py --name 'CASCADE+CSPDarknet' --config_path '/opt/ml/baseline/mmdetection/configs/_trash_/dark.py'
python train_base.py --name 'CASCADE+swin' --config_path '/opt/ml/baseline/mmdetection/configs/_trash_/swin.py'
python train_base.py --name 'CASCADE+CSPDarknet'
python train_base.py --name 'CASCADE+CSPDarknet'

# output1=$(python print_key.py)
# python tools/test.py $output1

# output2=$(python ./print_confusion.py)
# python tools/analysis_tools/confusion_matrix.py $output2

