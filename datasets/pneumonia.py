import os 
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase


@DATASET_REGISTRY.register()
class Pneumonia(DatasetBase):
    
    dataset_dir = "Pneumonia"  
    
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))  # NOTE: please modify root at train_script.sh
        self.dataset_dir = os.path.join(root, self.dataset_dir + "/chest_xray")
        
        self.image_dir_train = os.path.join(self.dataset_dir, "train")
        self.image_dir_val = os.path.join(self.dataset_dir, "val")
        self.image_dir_test = os.path.join(self.dataset_dir, "test")

        self.all_class_names = ["normal", "pneumonia"]

        train = self.read_data(split="train")
        val = self.read_data(split="val")
        test = self.read_data(split="test")  

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self, split):
        items = []
            
        if split == "train":
            normal_dir = os.path.join(self.image_dir_train, "NORMAL")
            pneumonia_dir = os.path.join(self.image_dir_train, "PNEUMONIA")
        elif split == "val":
            normal_dir = os.path.join(self.image_dir_val, "NORMAL")
            pneumonia_dir = os.path.join(self.image_dir_val, "PNEUMONIA")
        else:
            normal_dir = os.path.join(self.image_dir_test, "NORMAL")
            pneumonia_dir = os.path.join(self.image_dir_test, "PNEUMONIA")

        normal_images = os.listdir(normal_dir)
        for image in normal_images:
            assert image[-5:] == ".jpeg", f"the file {image} is not in .jpeg format"
            impath = os.path.join(normal_dir, image)
            label = 0
            cls_name = "normal"
            item = Datum(impath=impath, label=label, classname=cls_name)
            items.append(item)

        pneumonia_images = os.listdir(pneumonia_dir)
        for image in pneumonia_images:
            assert image[-5:] == ".jpeg", f"the file {image} is not in .jpeg format"
            impath = os.path.join(pneumonia_dir, image)
            label = 1
            cls_name = "pneumonia"
            item = Datum(impath=impath, label=label, classname=cls_name)
            items.append(item)
        
        return items
        