from DCDM_Train import train, eval
import argparse

def main(state='train',
         device = 'cuda:0',
         label_id = None,
         PCA_FCEL=False,
         AEN=False,
         epoch=200,
         data_dir='data/MT/train',
         save_weight_dir = 'models',
         save_F_datadir='data/MT/F_train',
         g_data_dir="data/MT/GData"):
    modelConfig = {
        "state": state, # or eval
        "epoch": epoch,
        "batch_size": 2,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "PCA_FCEL": PCA_FCEL,
        "AEN":AEN,
        'embedding_type': 1,
        "attn": [],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 64,
        "grad_clip": 1.,
        "device": device, ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_weight_dir": save_weight_dir,
        "g_data_dir": g_data_dir,
        "test_load_weight": "ckpt_199_.pt",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs3.png",
        "sampledImgName": "SampledNoGuidenceImgs3.png",
        "nrow": 9,
        "data_dir": data_dir,
        "save_F_datadir": save_F_datadir,
        "num_labels":6,
        "num_shapes":17,
        "w": 0.5,
        'label_id': label_id,
        'repeat': 1
        }
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    def args_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--state', default='test', type=str, help='train/test')
        parser.add_argument('--device', default='cuda:0', type=str, help='device')
        parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
        parser.add_argument('--train_datadir', type=str, default='/home/chase/shy/FL-DiG/data/NEU-CLS/train', help='datadir of train dataset')
        parser.add_argument('--save_F_datadir', type=str, default='/home/chase/shy/FL-DiG/data/NEU-CLS/F_train', help='save datadir of PCA_FCEL')
        parser.add_argument('--save_weight_dir', type=str, default='/home/chase/shy/dataset/FL-DiG_DiffModel/test2/NEU_model_epoch200_T1000__att0', help='save weight dir')
        parser.add_argument('--g_data_dir', default="/home/chase/shy/FL-DiG/data/NEU-CLS/GData", help='save datadir of genrated data')
        parser.add_argument('--PCA_FCEL', type=bool, default=True, help='align process')
        parser.add_argument('--AEN', type=bool, default=True, help='add fedprox proximal_term')
        parser.add_argument('--label_id', default=None, help='genration label id')


        args = parser.parse_args()
        return args
    args = args_parser()
    main(state=args.state,
         device=args.device,
         label_id=args.label_id,
         PCA_FCEL=args.PCA_FCEL,
         AEN=args.AEN,
         epoch=args.epochs,
         data_dir=args.train_datadir,
         save_weight_dir=args.save_weight_dir,
         save_F_datadir=args.save_F_datadir,
         g_data_dir=args.g_data_dir)
    # if state == 'train':
    #     main(state=state)
    # else:
    #     main(state=state)
        # for i in range(1):
        #     t = threading.Thread(target=main, args=(state, 'cuda:'+str(1), i))
        #     t.start()
        # t.join()
        # main(None, 'cuda:'+str(i), i)
