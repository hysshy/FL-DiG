from Train import train, eval
import threading

def main(state='train', device = 'cuda:0', label_id = None):
    modelConfig = {
        "state": state, # or eval
        "epoch": 200,
        "batch_size": 5,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
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
        "save_weight_dir": "./test/0221_1/",
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./test/0221_1/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs4.png",
        "sampledImgName": "SampledNoGuidenceImgs4.png",
        "nrow": 8,
        "data_dir": '/home/chase/shy/DDPM4MINER/data/ddpm_miner',
        "num_labels":7,
        "num_shapes":13,
        'embedding_type':1,
        "w": 0.2,
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
        for i in range(2):
            t = threading.Thread(target=main, args=(state, 'cuda:'+str(i), i))
            t.start()
        t.join()
        # main(None, 'cuda:'+str(i), i)
