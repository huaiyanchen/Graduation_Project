from __future__ import print_function

import torch
import json
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import sys,os,dlib,glob,numpy
from skimage import io
import os,sys,cv2,random,datetime,time,math
import argparse
import numpy as np

import net_s3fd
from bbox import *

import sys
from PyQt5 import QtCore, QtWidgets, uic
import os
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsScene, QGraphicsPixmapItem, QFileDialog
from PyQt5.QtCore import Qt
qtCreatorFile = "face.ui"  # Enter file here.

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

def detect(net,img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
    BB,CC,HH,WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(int(len(olist)/2)): olist[i*2] = F.softmax(olist[i*2])
    for i in range(int(len(olist)/2)):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, net, args, predictor_path,face_rec_model_path, faces_folder_path):

        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.descriptors = []

        for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
            print("Processing file: {}".format(f))
            img = io.imread(f)
            dets = self.detector(img, 1)
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):  
                shape = self.sp(img, d)
                face_descriptor = self.facerec.compute_face_descriptor(img, shape)
                v = numpy.array(face_descriptor)  
                self.descriptors.append(v)
        self.candidate = ['chy','shy','Shishi','Unknown4','Bingbing','Feifei']
        
        self.net = net
        self.args = args

        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.image1 = QPixmap()
        self.image2 = QPixmap()
        self.show_()

    def detection(self):
        if self.args.path=='CAMERA': cap = cv2.VideoCapture(0)
        if self.args.path=='CAMERA': ret, img = cap.read()
        else: img = cv2.imread(self.args.path)

        imgshow = np.copy(img)
        cv2.imwrite('E:/photoface/dlib/img_origin.jpg', imgshow)
        bboxlist = detect(self.net,img)

        keep = nms(bboxlist,0.3)
        bboxlist = bboxlist[keep,:]
        cnt = 0
        bboxes = []
        for b in bboxlist:
            x1,y1,x2,y2,s = b
            if s<0.5: continue
            result = {}
            result['x1'] = x1.item()
            result['x2'] = x2.item()
            result['y1'] = y1.item()
            result['y2'] = y2.item()
            bboxes.append(result)
            cv2.imwrite('E:/photoface/dlib/test'+str(cnt)+'.jpg', imgshow[int(x1):int(x2),int(y1):int(y2),:])
            cnt += 1
        return bboxes

    def recognition(self, path):
        img = io.imread(path)
        dets = self.detector(img, 1)
        dist = []
        for k, d in enumerate(dets):
            shape = self.sp(img, d)
            face_descriptor = self.facerec.compute_face_descriptor(img, shape)
            d_test = numpy.array(face_descriptor) 
            for i in self.descriptors:
                dist_ = numpy.linalg.norm(i-d_test)
                dist.append(dist_)

        c_d = dict(zip(self.candidate,dist))

        cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
        return cd_sorted[0][0]
        
    def show_(self):
        while(True):
            bboxes = self.detection()
            path1 = 'E:/photoface/dlib/img_origin.jpg'
            path2 = 'E:/photoface/dlib/img_reg.jpg'
            for i in range(len(bboxes)):
                img = cv2.imread(path1)
                x1 = bboxes[i]['x1']
                x2 = bboxes[i]['x2']
                y1 = bboxes[i]['y1']
                y2 = bboxes[i]['y2']
                name = self.recognition('E:/photoface/dlib/img_' + str(i) + '.jpg')
                cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1)
                cv2.putText(img, name, (int(x1),int(y1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                cv2.imwrite(path2, img)
            
            self.image1.load(path1)
            self.image2.load(path2)
            self.image1 = self.image1.scaled(800, 500, aspectRatioMode = Qt.KeepAspectRatio)
            self.image2 = self.image2.scaled(800, 500, aspectRatioMode=Qt.KeepAspectRatio)
            self.img1.scene = QGraphicsScene()
            item = QGraphicsPixmapItem(self.image1)
            self.img1.scene.addItem(item)
            self.img1.setScene(self.img1.scene)

            self.img2.scene = QGraphicsScene()
            item = QGraphicsPixmapItem(self.image2)
            self.img2.scene.addItem(item)
            self.img2.setScene(self.img2.scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch face detect')
    parser.add_argument('--net','-n', default='s3fd', type=str)
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--path', default='CAMERA', type=str)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    net = getattr(net_s3fd,args.net)()
    if args.model!='' :net.load_state_dict(torch.load(args.model))
    else: print('Please set --model parameter!')
    net.cuda()
    net.eval()

    app = QtWidgets.QApplication(sys.argv)
    window = MyApp(net, args)
    window.show()
    sys.exit(app.exec_())
    app.f.close()