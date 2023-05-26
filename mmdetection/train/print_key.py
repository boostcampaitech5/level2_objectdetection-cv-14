# if __name__ == '__main__':
#     from main_file import args
#     print(args.config_path,f'/opt/ml/baseline/mmdetection/work_dirs/{args.name}/latest.pth --out /opt/ml/baseline/mmdetection/work_dirs/{args.name}/test.pkl')

if __name__ == '__main__':
    from main_file import args
    print(args.config_path,f'/opt/ml/baseline/mmdetection/work_dirs/Cascade_RCNN+ConvNext_xlarge/epoch_36.pth --out /opt/ml/baseline/mmdetection/work_dirs/Cascade_RCNN+ConvNext_xlarge/test.pkl')
    