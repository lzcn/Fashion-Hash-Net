import os
import pickle

import PIL

from torchvision import transforms


# TODO: Check transforms
def _load_s2v(root="data/polyvore/sentence_vector"):
    # load normalized features of sentences
    data_fn = os.path.join(root, "semantic.pkl")
    with open(data_fn, "rb") as f:
        s2v = pickle.load(f)
    return s2v


class ResizeToSquare(object):
    def __init__(self, size):
        self.s = size
        self.size = (size, size)

    def __call__(self, im):
        if im.size == self.size:
            return im
        w, h = im.size
        ratio = self.s / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        im = im.resize((new_w, new_h), PIL.Image.BILINEAR)
        new_im = PIL.Image.new("RGB", self.size, (255, 255, 255))
        new_im.paste(im, ((self.s - new_w) // 2, (self.s - new_h) // 2))
        return new_im

    def __repr__(self):
        return self.__class__.__name__ + "(size={0})".format(self.s)


def get_img_trans(phase, image_size=291):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if phase == "train":
        if image_size == 291:
            return transforms.Compose(
                [transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif phase in ["test", "val"]:
        if image_size == 291:
            return transforms.Compose([transforms.ToTensor(), normalize])
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise KeyError
