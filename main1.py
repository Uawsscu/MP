############################################# IMPORT Tensorflow #########################################################


import argparse as ap
import cv2
import imutils
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
import freenect

#image_path = "/home/uawsscu/PycharmProjects/sift3/B-O-W/dataset/test/dall/dall7.jpg"


import numpy as np
import os
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#-----------------Model-----------------

import argparse as ap
import cv2
import imutils
import numpy as np
import os

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler


############################################  IMPORT Sphinx  ########################################################


from os import path
import pyaudio
import time
import Tkinter as tk
import os
from pocketsphinx.pocketsphinx import *
from sphinxbase.sphinxbase import *

from check4 import *
from manageDB import *



MODELDIR = "/home/mprang/PycharmProjects/object_detection/object_recognition_detection/model_LG"
DATADIR = "/home/mprang/PycharmProjects/object_detection/object_recognition_detection/dataLG"

config = Decoder.default_config()
config.set_string('-logfn', '/dev/null')
config.set_string('-hmm', path.join(MODELDIR, 'en-us/en-us'))
config.set_string('-lm', path.join(MODELDIR, 'en-us/en-us.lm.bin'))
config.set_string('-dict', path.join(MODELDIR, 'en-us/cmudict-en-us.dict'))
decoder = Decoder(config)

# Switch to JSGF grammar
jsgf = Jsgf(path.join(DATADIR, 'sentence.gram'))
rule = jsgf.get_rule('sentence.move') #>> public <move>
fsg = jsgf.build_fsg(rule, decoder.get_logmath(), 7.5)
fsg.writefile('sentence.fsg')

decoder.set_fsg("sentence", fsg)
decoder.set_search("sentence")

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
stream.start_stream()

in_speech_bf = False
decoder.start_utt()

STPindex = 0
STPname =""


####################################################################################################
#################################  DETECTIONS && PREDICTION ########################################

def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


def detectBOW():
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 20  # 20 S.
    clf, classes_names, stdSlr, k, voc = joblib.load("train.pkl")
    print "Ready!! Yessss"
    cap = cv2.VideoCapture(1)
    x = y = xh = yh = 1
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            while (ret):
                elapsed = int(time.time() - start)
                ret, image_np = cap.read()
                #image_np = get_video()
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)

                ####_______________________________________######

                fea_det = cv2.FeatureDetector_create("SIFT")
                des_ext = cv2.DescriptorExtractor_create("SIFT")

                des_list = []
                try:
                    y = int(vis_util.f.getYmin() * 479.000)
                    yh = int(vis_util.f.getYmax() * 479.000)
                    x = int(vis_util.f.getXmin() * 639.000)
                    xh = int(vis_util.f.getXmax() * 639.000)
                    im = image_np[y:yh, x:xh]
                    kpts = fea_det.detect(im)
                    kpts, des = des_ext.compute(im, kpts)
                    des_list.append((im, des))

                    descriptors = des_list[0][1]
                    for image2, descriptor in des_list[0:]:
                        descriptors = np.vstack((descriptors, descriptor))

                    test_features = np.zeros((1, k), "float32")
                    for i in xrange(1):
                        words, distance = vq(des_list[i][1], voc)
                        for w in words:
                            test_features[i][w] += 1

                    nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
                    idf = np.array(np.log((1.0 * 1 + 1) / (1.0 * nbr_occurences + 1)), 'float32')

                    test_features = stdSlr.transform(test_features)

                    # print clf.predict(test_features) >>>>>> class n. [0] [1]...[n]
                    predictions = [classes_names[i] for i in clf.predict(test_features)]

                    cv2.putText(image_np, str(predictions[0]), (x, y), font, 1, (0, 255, 255), 2)
                    # print predictions[0]

                except:
                    print('..')

                    # cv2.putText(image, prediction, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
                if (elapsed >= seconds):
                    cv2.destroyAllWindows()
                    return str(predictions[0])

                cv2.imshow('image', cv2.resize(image_np, (640, 480)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


#################################### CAP IMAGE ###############################################


def capture(namePath,obj_name):
    print "CAPPP"
    import time
    start = time.time()
    time.clock()
    elapsed = 0
    seconds = 200  # 20 S.
    cap = cv2.VideoCapture(1)
    # Running the tensorflow session
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            ret = True
            while (ret):
                ret, image_np = cap.read()
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                elapsed = int(time.time() - start)
                print "EP : ", elapsed

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)

                cv2.imshow('image', cv2.resize(image_np, (640, 480)))

                if (elapsed % 10 == 0):

                    try:
                        y = int(vis_util.f.getYmin() * 479.000)
                        yh = int(vis_util.f.getYmax() * 479.000)
                        x = int(vis_util.f.getXmin() * 639.000)
                        xh = int(vis_util.f.getXmax() * 639.000)
                        print y, " ", yh, " ", x, " ", xh
                        cv2.imshow('RGB image', image_np)

                        params = list()
                        # 143 : 869 // 354 :588
                        # 120:420, 213:456
                        crop_img = image_np[y:yh, x:xh]

                        cv2.imwrite(namePath + obj_name + str(
                            elapsed / 10) + ".jpg",
                                    crop_img, params)
                        print "OK cap"
                        cv2.destroyAllWindows()
                    except:
                        print "no image PASS"

                if (elapsed >= seconds):
                    cv2.destroyAllWindows()
                    break

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
    cap.release()
    cv2.destroyAllWindows()


########################################################################################################
########################################## SAVE MODEL ##################################################

def save_model():
    train_path = "/home/mprang/PycharmProjects/object_detection/object_recognition_detection/pic"
    training_names = os.listdir(train_path)
    image_paths = []
    image_classes = []  ## 00000,111111,2222,33333
    class_id = 0
    for training_name in training_names:
        dir = os.path.join(train_path, training_name)
        class_path = imutils.imlist(dir)

        image_paths += class_path
        image_classes += [class_id] * len(class_path)
        class_id += 1

    # Create feature extraction and keypoint detector objects
    # print image_classes," imP :",image_paths

    fea_det = cv2.FeatureDetector_create("SIFT")
    des_ext = cv2.DescriptorExtractor_create("SIFT")

    # List where all the descriptors are stored
    des_list = []

    for image_path in image_paths:
        # print image_path
        im = cv2.imread(image_path)
        kpts = fea_det.detect(im)
        kpts, des = des_ext.compute(im, kpts)
        des_list.append((image_path, des))

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0][1]
    for image_path, descriptor in des_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))

    # Perform k-means clustering
    k = 100
    voc, variance = kmeans(descriptors, k, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(image_paths), k), "float32")  # len(ALL pic) >> [0000000][00000]...

    for i in xrange(len(image_paths)):
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

    # Scaling the words
    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    # Train the Linear SVM
    clf = LinearSVC()
    clf.fit(im_features, np.array(image_classes))
    # Save the SVM

    joblib.dump((clf, training_names, stdSlr, k, voc), "train.pkl", compress=3)
    print "SAVE MODEL"
###########################################################################################################
import sys
import rospy
import sqlite3
import time
from std_msgs.msg import UInt8
from std_msgs.msg import String
from std_msgs.msg import UInt16


def callback_Talk(msg):  # insert ActionName
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute('SELECT * FROM ActionName')
            rows = cur.fetchall()
            lenR = len(rows)
            cur.execute('insert into ActionName (Name,ID) values (?,?)', (msg, lenR + 1,))
        except:
            return "Name Error"


def selectID_AcName(Action):
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("Select ID from ActionName where Name = ?", (Action,))
            row = cur.fetchone()
            for element in row:
                id = element
                return id
        except:
            return "Error"


def select_Buffer():
    list = []
    with sqlite3.connect("Test_PJ2.db") as con:
        cur = con.cursor()
        try:
            cur.execute("Select ID,M1,M2,M3,M4,M5,M6,M7,M8 from Buffer_Action")
            row = cur.fetchall()
            for element in row:
                motor = element
                list.append(motor)
            return list
        except:
            return "Null"


def del_buff():
    with sqlite3.connect("Test_PJ2.db") as con:
        sql_cmd = """
        delete from Buffer_Action

        """
        con.execute(sql_cmd)


def talker(msg):
    pub1 = rospy.Publisher('setMotor', UInt16, queue_size=10)

    rospy.init_node('Talker', anonymous=True)

    pub1.publish(int(msg))


def talker1(msg1):
    pub2 = rospy.Publisher('joints', String, queue_size=10)

    rospy.init_node('Talker', anonymous=True)

    pub2.publish(String(msg1))


################################################### MAIN ##################################################

JOB = True
JOB_HowTo_Open = False
STPindex = 0
jerry = True

while True:

    buf = stream.read(1024)
    if buf:
        decoder.process_raw(buf, False, False)

        if decoder.get_in_speech() != in_speech_bf:
            in_speech_bf = decoder.get_in_speech()
            if not in_speech_bf:
                decoder.end_utt()


                try:
                    strDecode = decoder.hyp().hypstr
                    #print strDecode

                    if strDecode != '':
                        #print strDecode
                        # >>>>>>> END <<<<<<<<<<<<
                        if JOB == True and strDecode[-3:] == 'end' and strDecode[:9] == "this is a" :
                            JOB = False
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode

                            obj_name = get_object_train(strDecode)  # sentence to word
                            print "Speech : ", obj_name
                            # create folder
                            dataset_Path = r'/home/mprang/PycharmProjects/object_detection/object_recognition_detection/pic/' + obj_name
                            p="/home/mprang/PycharmProjects/object_detection/object_recognition_detection/pic/"+ obj_name+"/"

                            if not os.path.exists(dataset_Path):
                                print dataset_Path
                                os.makedirs(dataset_Path)
                                capture(p,obj_name)  #capture image for train >> SAVE IMAGE
                                lenObj = int(lenDB("Corpus_Main.db", "SELECT * FROM obj_ALL"))  # count ROWs
                                insert_object_Train(obj_name, int(lenObj + 1))  # check Found objects?
                            JOB = True
                            save_model()


                        # >>>>>>> ARM <<<<<<<<<<<<
                        elif JOB ==True and strDecode[:14] == 'this is how to':
                            JOB = False
                            JOB_HowTo_Open = True
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode

                            STPname = get_V(strDecode)  # grab
                            #insert table
                            try :
                                with sqlite3.connect("Test_PJ2.db") as con:
                                    cur = con.cursor()
                                    lenObj = int(lenDB("Test_PJ2.db", "SELECT * FROM ActionName"))  # count ROWs
                                    cur.execute('insert into ActionName (ID,Name) values (?,?)', (lenObj + 1, STPname))
                                    print(STPname)
                            except :
                                print "Action in table!!!"
                            print("SAVE NAME TO Table Main_action")


                        elif JOB_HowTo_Open == True and strDecode == 'call back step':
                            print 'Stream decoding result:', strDecode
                            STPindex +=1
                            print STPindex, " : ", STPname
                            talker(9)
                            #SAVE Action
                        elif JOB_HowTo_Open == True and strDecode == 'stop call back':
                            JOB = True
                            JOB_HowTo_Open = False
                            STPindex = 0
                            list = []
                            for i in select_Buffer():
                                list.append(selectID_AcName(STPname))
                                for x in i:
                                    list.append(x)
                                with sqlite3.connect("Test_PJ2.db") as con:
                                    cur = con.cursor()
                                    cur.execute(
                                        'insert into Action_Robot (ID,StepAction,M1,M2,M3,M4,M5,M6,M7,M8) values (?,?,?,?,?,?,?,?,?,?)',
                                        (list))
                                    print(list)
                                    del list[:]
                            del_buff()
                            print "STOP.."

                        # >>>>>>> JERRY <<<<<<<<<<<<
                        elif strDecode == 'stop action':
                            jerry == False

                        elif strDecode[:5] == 'jerry':
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode
                            v = get_V(strDecode)#
                            with sqlite3.connect("Test_PJ2.db") as con:
                                cur = con.cursor()
                                cur.execute(
                                    'Select Action_Robot.M1,Action_Robot.M2,Action_Robot.M3,Action_Robot.M4,Action_Robot.M5,Action_Robot.M6,Action_Robot.M7,Action_Robot.M8 from Action_Robot inner join ActionName on Action_Robot.ID = ActionName.ID where Name = ?',
                                    (v,))
                                row = cur.fetchall()



                                for element in row:
                                    joint2 = str(element)
                                    # command1 = command1 + joint2
                                    command2 = joint2[1:]
                                    print(command2)
                                    if jerry == False:
                                        jerry = True
                                        break

                                    talker1(command2)
                                    # joint = ""
                                    command1 = ""

                                    time.sleep(3)
                            #corpus_Arm


                        # >>>>>>> PASS DO YOU KNOW~??? <<<<<<<<<<<<
                        elif JOB == True and strDecode[:11] == 'do you know':
                            JOB = False
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode
                            obj_name = get_object_question(strDecode)
                            print(obj_name)
                            obj_find = search_object_Train(obj_name)

                            if obj_find != "None":
                                print "Yes , I know!"
                            else: print "No , I don't know!"
                            JOB = True
                        elif JOB == True and strDecode[:22]== 'hey jerry what is that':
                            print "\n------------------------------------------"
                            print '\nStream decoding result:', strDecode
                            obj_detect = detectBOW()
                            print "That is a ",obj_detect


                except AttributeError:
                    pass
                decoder.start_utt()
    else:
        break
decoder.end_utt()
print('An Error occured :', decoder.hyp().hypstr)

print "OK"