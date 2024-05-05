import json

from torchvision.datasets import CocoDetection


class CustomCocoDataset(CocoDetection):

    def __init__(self, root: str, annotation_file: str, make_index: bool, transform=None):
        super().__init__(root, annotation_file, transform)

        self.annotations = dict()
        self.index = dict()
        self.make_index = make_index
        if self.make_index:
            self.__create_index__(annotation_file)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __create_index__(self, annotation_file: str):
        with open(annotation_file) as f:
            self.annotations = json.load(f)
            for idx in range(len(self.annotations["images"])):
                img = self.annotations["images"][idx]
                self.index[img["id"]] = [idx, img["file_name"]]

    def get_filename_by_id(self, img_id) -> str:
        if not self.make_index:
            return ""
        return self.index[img_id][1]

    def update_annotations(self, img_id, kv: dict):
        if not self.make_index:
            return
        index = self.index[img_id]
        idx = index[0]
        for k, v in kv.items():
            self.annotations["images"][idx][k] = v

    def save_annotations(self, annotation_file: str):
        if not self.make_index:
            return
        with open(annotation_file, "w") as f:
            json.dump(self.annotations, f)
