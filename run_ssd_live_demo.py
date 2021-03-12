import threading

import torch

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite, \
    create_custom_mobilenetv3_small_ssd_lite
from vision.utils.misc import Timer
import cv2
import sys

if len(sys.argv) < 4:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> [video file]')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    if 'rtsp' not in sys.argv[4] and 'file' not in sys.argv[4] and 'http' not in sys.argv[4]:
        raise RuntimeError("잘못된 주소")
    cap = cv2.VideoCapture(sys.argv[4], cv2.CAP_FFMPEG)  # capture from file
else:
    cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

# for realtime image

# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_ret = None
        self.last_frame = None
        self.running = False
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        self.running = True
        while self.running:
            self.last_ret, self.last_frame = self.camera.read()

    def close(self):
        self.running = False
        self.join()

    def status(self):
        return self.last_ret

cap_cleaner = CameraBufferCleanerThread(cap)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)


if net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'custom-mb3-small-ssd-lite':
    net = create_custom_mobilenetv3_small_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of mb3-large-ssd-lite, mb3-small-ssd-lite and custom-mb3-small-ssd-lite.")
    sys.exit(1)
net.load(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=torch.device('cuda:0'))

timer = Timer()
while True:
    orig_image = cap_cleaner.last_frame
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        box = [int(x) for x in box]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

        cv2.rectangle(orig_image, (box[0], box[1], box[2] - box[0], box[3] - box[1]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Waiting for cleaner to exit ...")
cap_cleaner.close()
cap.release()
cv2.destroyAllWindows()
