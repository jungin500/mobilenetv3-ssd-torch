from ..transforms.transforms import *


def TrainAugFinalComputation(img, boxes=None, labels=None):
    return img / 128.0, boxes, labels

def TestTrfmFinalComputation(img, boxes=None, labels=None):
    return img / 128.0, boxes, labels

def PredTrfmFinalComputation(img, boxes=None, labels=None):
    return img / 128.0, boxes, labels

class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """

        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            TrainAugFinalComputation,
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):

        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            TestTrfmFinalComputation,
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):

        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            PredTrfmFinalComputation,
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image