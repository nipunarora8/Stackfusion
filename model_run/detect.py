import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import model_run.core.utils as utils
from model_run.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import easyocr

# flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
# flags.DEFINE_string('output', 'result.png', 'path to output image')

def main(file_path):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    input_size = 416
    # image_path = FLAGS.image

    img = cv2.imread(file_path)
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (416, 416))
    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    saved_model_loaded = tf.saved_model.load('model_run/anpr_yolo_tf', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.60,
        score_threshold=0.60
    )
    
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    image, plate = utils.draw_bbox(original_image, pred_bbox)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(plate)

    for i in result:
        result_str =i[1] +" "

    return image, plate, result_str

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
