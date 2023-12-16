import argparse

def read_model_args():
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', default=30, type=int, help=""" 训练批次 """)
    args.add_argument('--model_dir', required=True, help=""" bert模型路径，必须给定 """)
    args.add_argument('--batch_size', default=64, type=int, help=""" 训练batch的大小 """)
    args.add_argument('--output_dir', default=r'model/datas/model', help=""" 模型保存路径 """)
    args.add_argument('--labels_map', default=r'datas/labels_map.json', help=""" 标签文件 """)
    args.add_argument('--id2labels', default=r'datas/id2labels_map.json', help=""" id到label的映射""")
    args.add_argument('--save_size', default=10, type=int, help=""" 模型每隔多少epoch保存一次 """)
    args.add_argument('--labels_num', default=12, type=int, help=""" 标签数量 """)
    args.add_argument('--num_workers', default=0,type=int, help=""" 处理数据的进程数 """)
    args.add_argument('--shuffle', default=True, help=""" 分割数据时是否进行打乱 """)
    args.add_argument(
        '--train_data_path', default=r'datas/split_train_data.csv',
        help=""" 训练数据路径，默认是datas路径下的split_train_data.csv"""
    )
    args.add_argument(
        '--test_data_path', default=r'datas/split_val_data.csv',
        help=""" 训练数据路径，默认是datas路径下的split_test_data.csv"""
    )
    return args


