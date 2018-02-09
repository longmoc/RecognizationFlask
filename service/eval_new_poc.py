import tensorflow as tf
from PIL import Image
from nets.inception_resnet_v2 import *
import numpy as np
from datetime import datetime
import cv2
import os
from flask import request, Flask, render_template, Response

from functools import wraps

slim = tf.contrib.slim

USERNAME = 'pocadmin'
PASSWORD = 'pocpasswd'
labels = ['', 'OK', 'NG']

UPLOAD_FOLDER = 'tmp/'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'bmp'])
app = Flask(__name__, static_folder="tmp")
app.config.update(
    WTF_CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess'
)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

tf.app.flags.DEFINE_string('checkpoint', '', 'Checkpoint path')
tf.app.flags.DEFINE_integer('port', 5000, 'Site port')
tf.app.flags.DEFINE_integer('device', 1, 'Running device')
tf.app.flags.DEFINE_float('gpu_mem_fraction', 0.5, 'Per process GPU memory fraction')
FLAGS = tf.app.flags.FLAGS

input_tensor = tf.placeholder(tf.float32, shape=(None, 299, 299, 3), name='input_image')
scaled_input_tensor = tf.scalar_mul((1.0 / 255), input_tensor)
scaled_input_tensor = tf.subtract(scaled_input_tensor, 0.5)
scaled_input_tensor = tf.multiply(scaled_input_tensor, 2.0)


def check_auth(username, password):
    return username == USERNAME and password == PASSWORD


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[-1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
@app.route('/upload')
@app.route('/favicon.ico')
@requires_auth
def upload_file():
    return render_template('upload.html')


@requires_auth
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        f = request.files['file']
        if not allowed_file(f.filename):
            return 'Your file is not allowed by this service. Allowed extensions: .jpg, .JPG, .jpeg, .JPEG, .png, .PNG, .bmp, .BMP'
        cur_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S%f')
        ext = f.filename.rsplit('.', 1)[-1].lower()
        new_name = '{}.{}'.format(cur_time, ext)
        uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
        f.save(uploaded_file_path)
        edge_img, label, pred = evaluate(uploaded_file_path)
        print(uploaded_file_path, label, pred)
        return render_template("detect.html", title='Result', image_raw=new_name,
                               image_edge=edge_img, label=label, pred=pred * 100, image_name=f.filename)


def evaluate(image_path):
    edge_path = line_detect(image_path)
    im = Image.open(edge_path).resize((299, 299))
    im = np.array(im)
    im = im.reshape(-1, 299, 299, 3)
    predict_values, logit_values = sess.run([end_points['Predictions'], logits], feed_dict={input_tensor: im})
    edge_path = edge_path.split('/')[-1].split('\\')[-1]
    label = labels[int(np.argmax(predict_values))]
    return edge_path, label, np.max(predict_values)


def line_detect(image_path):
    in_img = cv2.imread(image_path)
    gray = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(gauss, 100, 150)
    LSD = cv2.createLineSegmentDetector()
    lines, width, prec, nfa = LSD.detect(edges)
    edge_path = image_path.split('.')[0] + '_cropped.jpg'

    x_min = 10e6
    y_min = 10e6
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            x_min = min([x1, x2, x_min])
            y_min = min([y1, y2, y_min])

    x_min = int(x_min)
    y_min = int(y_min)
    result = in_img[y_min - 40:y_min + 400, x_min - 40:x_min + 520]
    cv2.imwrite(edge_path, result)
    return edge_path


if __name__ == '__main__':
    # Load the model
    gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_mem_fraction)
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': FLAGS.device}, gpu_options=gpu_option))
    arg_scope = inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_resnet_v2(scaled_input_tensor, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.checkpoint)
    app.run(debug=True, host='0.0.0.0', use_reloader=False, port=FLAGS.port)
