import numpy as np
#import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import argparse
import sys
import cv2
import math
rgb = True

def draw_bboxes (img, boxes, outname):
   for i in range(len(boxes)):
      xmin,ymin,xmax,ymax,cat = boxes[i]
      cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),4)
      font = cv2.FONT_HERSHEY_SIMPLEX
      #cv2.putText(img, str(cat), (xmin,ymin-10), font, 0.75, (255,255,255), 2, cv2.LINE_AA)
   print (outname)
   cv2.imwrite(outname, img) 

if __name__ == "__main__":
   #print ("Usage: python show_regions.py image.jpg boxes.txt image_with_boxes.jpg confidence_threshold")
   image = cv2.imread(sys.argv[1], int(rgb))
   fboxes = open(sys.argv[2], 'r')
   outname = sys.argv[3]
   confidence = float(sys.argv[4])
   boxes = []
   for line in fboxes:
      line = line.strip('\n')
      fields = line.split(' ')
      xmin = int(fields[0]) 
      ymin = int(fields[1]) 
      xmax = int(fields[2]) 
      ymax = int(fields[3])
      cat = int(fields[4])
      score = float(fields[5])
      #Showing only some categories:
      not_show = [44,50,51,54,55,59,72,73,74,75,76,77,79,82,83,84,86,89,91,93,94]
      if (score > confidence) and (cat not in not_show):
         boxes.append((xmin,ymin,xmax,ymax,cat))
   draw_bboxes(image, boxes, outname)
