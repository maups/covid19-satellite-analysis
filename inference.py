import PIL
import argparse
import numpy as np
import math
import tensorflow as tf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
from parameters import *

#--------------------------------------------------------------
def generate_detections(checkpoint, images):
    
   print("Creating Graph...")
   detection_graph = tf.Graph()
   with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(checkpoint, 'rb') as fid:
         serialized_graph = fid.read()
         od_graph_def.ParseFromString(serialized_graph)
         tf.import_graph_def(od_graph_def, name = '')

   boxes = []
   scores = []
   classes = []
   k = 0

   with detection_graph.as_default():
      with tf.Session(graph = detection_graph) as sess:
         for image_np in tqdm(images):
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            box = detection_graph.get_tensor_by_name('detection_boxes:0')
            score = detection_graph.get_tensor_by_name('detection_scores:0')
            clss = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (box, score, clss, num_detections) = sess.run(
                        [box, score, clss, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
            boxes.append(box)
            scores.append(score)
            classes.append(clss)
                
   boxes = np.squeeze(np.array(boxes))
   scores = np.squeeze(np.array(scores))
   classes = np.squeeze(np.array(classes))

   return boxes, scores, classes

#--------------------------------------------------------------
def split_image_with_overlap (image, chip_size=(300,300)):
   iw, ih, _ = image.shape
   wn, hn = chip_size
   wn_overlap = wn - number_of_overlapped_pixels
   hn_overlap = hn - number_of_overlapped_pixels
   slices_w = int(math.ceil(float(iw)/wn_overlap))
   slices_h = int(math.ceil(float(ih)/hn_overlap))
   shifts = []
   image_chunks = np.zeros((slices_w*slices_h, wn, hn, 3))
   index = 0
   for i in range(slices_w):
      for j in range(slices_h):
         if slices_w == 1 and slices_h == 1:
            chip = image[0 : iw, 0 : ih, : 3]
            shifts.append ((0,0))
         elif (i < (slices_w-1)) and (j < (slices_h-1)) and ((wn_overlap*(i+1))+number_of_overlapped_pixels < iw) and ((hn_overlap*(j+1))+number_of_overlapped_pixels < ih):
            chip = image[wn_overlap*i : (wn_overlap*(i+1))+number_of_overlapped_pixels, hn_overlap*j: (hn_overlap*(j+1))+number_of_overlapped_pixels, : 3]
            shifts.append ((wn_overlap*i,hn_overlap*j))
         elif (i < (slices_w-1)) and ((wn_overlap*(i+1))+number_of_overlapped_pixels < iw):
            hsidea = max(0, ih - hn)
            hsideb = ih
            chip = image[wn_overlap*i : (wn_overlap*(i+1))+number_of_overlapped_pixels, hsidea : hsideb, : 3]
            shifts.append ((wn_overlap*i,hsidea))
         elif j < (slices_h-1) and ((hn_overlap*(j+1))+number_of_overlapped_pixels < ih):
            wsidea = max(0, iw - wn)
            wsideb = iw
            chip = image[wsidea : wsideb, hn_overlap*j: (hn_overlap*(j+1))+number_of_overlapped_pixels, : 3]
            shifts.append ((wsidea,hn_overlap*j))
         else:
            hsidea = max(0, ih - hn)
            hsideb = ih
            wsidea = max(0, iw - wn)
            wsideb = iw
            chip = image[wsidea : wsideb, hsidea : hsideb, : 3]
            shifts.append ((wsidea,hsidea))
         image_chunks[index] = chip
         index += 1

   if verbose:
      print ('Number of slices (overlap): ', index)

   return image_chunks.astype(np.uint8), shifts

#--------------------------------------------------------------
def split_image (image, chip_size=(300,300)):
   iw, ih, _ = image.shape
   wn, hn = chip_size
   slices_w = int(math.ceil(float(iw)/wn))
   slices_h = int(math.ceil(float(ih)/hn))
   shifts = []
   image_chunks = np.zeros((slices_w*slices_h, wn, hn, 3))
   index = 0
   for i in range(slices_w):  
      for j in range(slices_h):
         if slices_w == 1 and slices_h == 1:
            chip = image[0 : iw, 0 : ih, : 3]
            shifts.append ((0,0))
         elif (i < (slices_w-1)) and (j < (slices_h-1)):
            chip = image[wn*i : wn*(i+1), hn*j : hn*(j+1), : 3]
            shifts.append ((wn*i,hn*j))
         elif i < (slices_w-1):
            hsidea = max(0, ih - hn)
            hsideb = ih
            chip = image[wn*i : wn*(i+1), hsidea : hsideb, : 3]
            shifts.append ((wn*i,hsidea))
         elif j < (slices_h-1):
            wsidea = max(0, iw - wn)
            wsideb = iw
            chip = image[wsidea : wsideb, hn*j : hn*(j+1), : 3]
            shifts.append ((wsidea,hn*j))
         else:
            hsidea = max(0, ih - hn)
            hsideb = ih
            wsidea = max(0, iw - wn)
            wsideb = iw
            chip = image[wsidea : wsideb, hsidea : hsideb, : 3]
            shifts.append ((wsidea,hsidea))
         image_chunks[index] = chip
         index += 1
    
   if verbose:
      print ('Number of slices (non-overlap): ', index)

   return image_chunks.astype(np.uint8), shifts

#--------------------------------------------------------------
def non_max_suppression_fast (boxes, overlapThresh):
        
   # if there are no boxes, return an empty list
   if len(boxes) == 0:
      return []

   # if the bounding boxes integers, convert them to floats --
   # this is important since we'll be doing a bunch of divisions
   if boxes.dtype.kind == "i":
      boxes = boxes.astype("float")

   # initialize the list of picked indexes 
   pick = []

   # grabbing the bounding boxes information:
   x1 = boxes[:,0]
   y1 = boxes[:,1]
   x2 = boxes[:,2]
   y2 = boxes[:,3]
   clas = boxes[:,4]
   score = boxes[:,5]

   # comute the area of the bounding boxes:
   area = (x2 - x1 + 1) * (y2 - y1 + 1)
   
   # NEW: Sorting by region confidence!!!     
   idxs = np.argsort(score) 

   # keep looping while some indexes still remain in the indexes list
   while len(idxs) > 0:
   
      # grab the last index in the indexes list and add the index value to the list of picked indexes
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = np.maximum(x1[i], x1[idxs[:last]])
      yy1 = np.maximum(y1[i], y1[idxs[:last]])
      xx2 = np.minimum(x2[i], x2[idxs[:last]])
      yy2 = np.minimum(y2[i], y2[idxs[:last]])
      cls = np.equal (clas[i], clas[idxs[:last]])
      cls = cls.astype(int)

      # compute the width and height of the bounding box
      w = np.maximum(0, xx2 - xx1 + 1)
      h = np.maximum(0, yy2 - yy1 + 1)

      # compute the ratio of overlap
      overlap1 = (w * h) / area[idxs[:last]]
      overlap2 = (w * h) / area[i]
      overlap = np.minimum (overlap1, overlap2)

      if merge_only_regions_from_same_class:
         overlap = overlap * cls

      if int(clas[i]) in large:
         overlapThresh = 0.5 

      indices_erased = np.where(overlap > overlapThresh)[0]

      if indices_erased != []:
                   
         x1_mean = x2_mean = y1_mean = y2_mean = weight_mean = 0.0
                   
         nsamples = 0
                   
         categories = []
                   
         categories.append(clas[i])
                   
         for index in indices_erased:
                       
            if score[idxs[index]] > threshold_for_roi_coords_update:

               if use_confidence_to_estimate_region_boundaries:                
                  x1_mean += (x1[idxs[index]] * score[idxs[index]])
                  y1_mean += (y1[idxs[index]] * score[idxs[index]])
                  x2_mean += (x2[idxs[index]] * score[idxs[index]])
                  y2_mean += (y2[idxs[index]] * score[idxs[index]])
                  weight_mean += score[idxs[index]]
               else:
                  x1_mean += (x1[idxs[index]])
                  y1_mean += (y1[idxs[index]])
                  x2_mean += (x2[idxs[index]])
                  y2_mean += (y2[idxs[index]])
              
               categories.append(clas[idxs[index]])
               nsamples += 1
                   
         if nsamples > 0:
                     
            if use_confidence_to_estimate_region_boundaries:                
               x1_mean = x1_mean/weight_mean
               y1_mean = y1_mean/weight_mean
               x2_mean = x2_mean/weight_mean
               y2_mean = y2_mean/weight_mean
            else:
               x1_mean = x1_mean/float(nsamples)
               y1_mean = y1_mean/float(nsamples)
               x2_mean = x2_mean/float(nsamples)
               y2_mean = y2_mean/float(nsamples)

            # Updating the boxes coordinates based on the region weights:
            boxes[i][0] = x1_mean
            boxes[i][1] = y1_mean
            boxes[i][2] = x2_mean
            boxes[i][3] = y2_mean
 
      # delete all indexes from the index list that have
      idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

   # return only the bounding boxes that were picked using the integer data type
   return boxes[pick]

#--------------------------------------------------------------
def process_image (scale, args, region_overlap, model_name, detector_name):

   inverse_scale = 1.0/scale

   #Parse and chip images
   image = Image.open(args.input)
   ow = float(image.size[0])
   oh = float(image.size[1])
   image = image.resize((int(scale * ow), int(scale * oh)), PIL.Image.ANTIALIAS)
   arr = np.array(image)

   if int(scale * ow) < args.chip_size or int(scale * oh) < args.chip_size:
       print ('Detection failed for model: ', detector_name, '. The image dimensions are to small!!')
       return np.squeeze(np.array([])), np.squeeze(np.array([])), np.squeeze(np.array([]))

   chip_size = (args.chip_size, args.chip_size)
       
   if region_overlap:
      images, shifts = split_image_with_overlap (arr, chip_size)
   else:
      images, shifts = split_image (arr, chip_size)

   #generate detections
   if model_name == "vanilla":
      boxes, scores, classes = generate_detections(args.checkpoint1, images)
   elif model_name == "multires":
      boxes, scores, classes = generate_detections(args.checkpoint2, images)
   else:
      print ('Error: choose a model!!!!') 

   #Process boxes to be full-sized
   width,height,_ = arr.shape
    
   cwn,chn = (chip_size)
  
   if region_overlap:
      wn_overlap = cwn - number_of_overlapped_pixels
      hn_overlap = chn - number_of_overlapped_pixels
      wn = int(math.ceil(float(width)/wn_overlap))
      hn = int(math.ceil(float(height)/hn_overlap))
   else:
      wn = int(math.ceil(float(width)/cwn))
      hn = int(math.ceil(float(height)/chn))

   num_preds = 250
   bfull = boxes[:wn*hn].reshape((wn, hn, num_preds, 4))
   b2 = np.zeros(bfull.shape)
   b2[:, :, :, 0] = bfull[:, :, :, 1]
   b2[:, :, :, 1] = bfull[:, :, :, 0]
   b2[:, :, :, 2] = bfull[:, :, :, 3]
   b2[:, :, :, 3] = bfull[:, :, :, 2]

   bfull = b2
   bfull[:, :, :, 0] *= cwn
   bfull[:, :, :, 2] *= cwn
   bfull[:, :, :, 1] *= chn
   bfull[:, :, :, 3] *= chn

   index = 0
   for i in range(wn):
      for j in range(hn):
         sx = shifts[index][1]
         sy = shifts[index][0]
         bfull[i, j, :, 0] = (sx + bfull[i, j, :, 0]) * inverse_scale
         bfull[i, j, :, 2] = (sx + bfull[i, j, :, 2]) * inverse_scale
         bfull[i, j, :, 1] = (sy + bfull[i, j, :, 1]) * inverse_scale
         bfull[i, j, :, 3] = (sy + bfull[i, j, :, 3]) * inverse_scale
         index += 1
            
   bfull = bfull.reshape((hn * wn, num_preds, 4))

   return bfull, scores, classes

#--------------------------------------------------------------
if __name__ == "__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("-c1","--checkpoint1", help = "Path to saved model")
   parser.add_argument("-c2","--checkpoint2", help = "Path to saved model")
   parser.add_argument("-cs", "--chip_size", default = 300, type = int, help = "Size in pixels to chip input image")
   parser.add_argument("-i", "--input", help = "Path to test chip")
   parser.add_argument("-o","--output", default = "predictions.txt", help = "Filepath of desired output")
   args = parser.parse_args()
  
   if model_one_activate:
      box1, cof1, cls1 = process_image (model_one_zoom, args, model_one_region_overlap, model_one_classifier, 'model_one')

   if model_two_activate:
      box2, cof2, cls2 = process_image (model_two_zoom, args, model_two_region_overlap, model_two_classifier, 'model_two')
   
   if model_three_activate:
      box3, cof3, cls3 = process_image (model_three_zoom, args, model_three_region_overlap, model_three_classifier, 'model_three')

   if model_four_activate:
      box4, cof4, cls4 = process_image (model_four_zoom, args, model_four_region_overlap, model_four_classifier, 'model_four')

   if model_five_activate:
      box5, cof5, cls5 = process_image (model_five_zoom, args, model_five_region_overlap, model_five_classifier, 'model_five')

   f = open(args.output,'w')

   region_list = []

   nof_candidate_regions = 0

   if model_one_activate:
      for i in range(box1.shape[0]):
         for j in range(box1[i].shape[0]):
            box = box1[i, j] #xmin ymin xmax ymax
            class_prediction = int(cls1[i, j])
            score_prediction = cof1[i, j]
            if ( (score_prediction > model_one_score_threshold) and (class_prediction in medium or class_prediction in small) ) or score_prediction > threshold_high_confidence:
               region_list.append([box[0], box[1], box[2], box[3], int(class_prediction), score_prediction])
               nof_candidate_regions += 1

   if model_two_activate:
      for i in range(box2.shape[0]):
         for j in range(box2[i].shape[0]):
            box = box2[i, j] #xmin ymin xmax ymax
            class_prediction = int(cls2[i, j])
            score_prediction = cof2[i, j]
            if ( (score_prediction > model_two_score_threshold) and (class_prediction in small) ) or ((score_prediction > threshold_high_confidence) and (class_prediction not in large)) or (class_prediction in medium and score_prediction > 0.15):
               region_list.append([box[0], box[1], box[2], box[3], int(class_prediction), score_prediction])
               nof_candidate_regions += 1

   if model_three_activate:
      for i in range(box3.shape[0]):
         for j in range(box3[i].shape[0]):
            box = box3[i, j] #xmin ymin xmax ymax
            class_prediction = int(cls3[i, j])
            score_prediction = cof3[i, j]
            if ((score_prediction > model_three_score_threshold) and (class_prediction in medium)) or ((class_prediction in large) and (score_prediction > 0.1)):
               region_list.append([box[0], box[1], box[2], box[3], int(class_prediction), score_prediction])
               nof_candidate_regions += 1

   if model_four_activate:
      for i in range(box4.shape[0]):
         for j in range(box4[i].shape[0]):
            box = box4[i, j] #xmin ymin xmax ymax
            class_prediction = int(cls4[i, j])
            score_prediction = cof4[i, j]
            if ((score_prediction > model_four_score_threshold) and (class_prediction in small)) or (class_prediction in large and score_prediction > 0.1) or (class_prediction in medium and score_prediction > 0.15):
               region_list.append([box[0], box[1], box[2], box[3], int(class_prediction), score_prediction])
               nof_candidate_regions += 1

   if model_five_activate:
      for i in range(box5.shape[0]):
         for j in range(box5[i].shape[0]):
            box = box5[i, j] #xmin ymin xmax ymax
            class_prediction = int(cls5[i, j])
            score_prediction = cof5[i, j]
            if (class_prediction in large and score_prediction > 0.3):
               region_list.append([box[0], box[1], box[2], box[3], int(class_prediction), score_prediction])
               nof_candidate_regions += 1

   region_list = np.array(region_list)

   region_list_after_nms = non_max_suppression_fast (region_list, percentage_of_overlap_to_merge)

   nof_detections = 0
   
   nof_regions_filtered_by_dimensions = 0

   for r in region_list_after_nms:
      xmin = int(r[0])
      ymin = int(r[1])
      xmax = int(r[2])
      ymax = int(r[3])
      clas = int(r[4])
      confidence = float(r[5])
      r_width  = xmax - xmin 
      r_height = ymax - ymin 
      if (r_width > min_region_width) and (r_height > min_region_height):
         f.write('%d %d %d %d %d %f \n' % (xmin, ymin, xmax, ymax, clas, confidence))
      else:
         nof_regions_filtered_by_dimensions += 1 
      nof_detections += 1

   # Avoiding empty files:
   if (nof_detections == 0):
      f.write (("%d %d %d %d %d %f\n") % (0, 0, 1, 1, 11, 0.001))

   if verbose:
      print ('# of candidates: ', nof_candidate_regions,', # of detections: ', nof_detections, ', # of regions filtered by dimensions: ', nof_regions_filtered_by_dimensions)

   f.close()
