from DCDM_Train import train, eval
import threading

def main(state='train', device = 'cuda:0', label_id = None):
    modelConfig = {
        "state": state, # or eval
        "epoch": 200,
        "batch_size": 2,
        "T": 700,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "PCA_FCEL": False,
        "AEN":True,
        'embedding_type': 1,
        "attn": [2,3],
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
        "save_weight_dir": "../test/NEU_model_epoch200_T700_imgsize64_embeddingtype1_fuzzy/",
        "test_load_weight": "ckpt_199_.pt",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs3.png",
        "sampledImgName": "SampledNoGuidenceImgs3.png",
        "nrow": 9,
        "data_dir": '/home/chase/shy/FedDGDA/data/MT/F-train',
        "num_labels":6,
        "num_shapes":17,
        "w": 1,
        'label_id': label_id,
        'repeat': 1
        }
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        eval(modelConfig)


if __name__ == '__main__':
    state = 'train'
    if state == 'train':
        main(state=state)
    else:
        main(state=state)
        # for i in range(1):
        #     t = threading.Thread(target=main, args=(state, 'cuda:'+str(1), i))
        #     t.start()
        # t.join()
        # main(None, 'cuda:'+str(i), i)
