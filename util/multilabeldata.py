import os
from torch.utils.data import Dataset
from PIL import Image

class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.label1list = os.listdir(root_dir)
        self.label2list = os.listdir(root_dir+'/'+self.label1list[0])

        for label1 in os.listdir(root_dir):
            for label2 in os.listdir(os.path.join(root_dir, label1)):
                for filename in os.listdir(os.path.join(root_dir, label1, label2)):
                    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_path = os.path.join(root_dir, label1, label2, filename)
                        self.samples.append((img_path, (self.label1list.index(label1), self.label2list.index(label2)+len(self.label1list)+1)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, labels


