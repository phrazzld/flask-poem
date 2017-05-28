# flask
from app import app
# sys nav and file mgmt
import os, sys
# brainssss
import tensorflow as tf
# fetch images
import requests
# for unique names
import random
# for ensuring our images are JPG
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def label(image_url):
    # download image from image_url
    r = requests.get(image_url)
    image_name = "image_to_classify__" + str(random.randint(1,10000)) + ".jpg"
    image_file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'user_photos', image_name), 'wb')
    for chunk in r.iter_content(100000):
        image_file.write(chunk)
    image_file.close()

    image_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'user_photos', image_name)

    # ensure our images are JPG
    im = Image.open(image_path)
    rgb_im = im.convert('RGB')
    rgb_im.save(image_path)

    # read the image data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # load label file, strip off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

    # unpersist graph from file
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        top_prediction = label_lines[top_k[0]]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            #print('%s (score = %.5f)' % (human_string, score))

        return top_prediction, '/static/user_photos/' + image_name
