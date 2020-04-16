#General parameter settings:

use_confidence_to_estimate_region_boundaries = True

min_region_width = 4 # in pixels!

min_region_height = 4 # in pixels!

percentage_of_overlap_to_merge = 0.55 #BOTH regions need to have this overlap percentage to be merged!!

threshold_high_confidence = 0.7 # Any region with this confidence score will be included even if its classification is small/medium or large set.

merge_only_regions_from_same_class = True

threshold_for_roi_coords_update = 0.05 #Regions with confidence higher than this threshold help to define the new region of interest (ROI). 

verbose = True

number_of_overlapped_pixels = 100 # For two adjacent regions!!!

#Model one settings:
model_one_activate = True
model_one_zoom = 1.0
model_one_region_overlap = False
model_one_score_threshold = 0.15
model_one_classifier = "vanilla"

#Model two settings:
model_two_activate = True
model_two_zoom = 1.3
model_two_region_overlap = False
model_two_score_threshold = 0.06
model_two_classifier = "vanilla"

#Model three settings:
model_three_activate = True
model_three_zoom = 0.7
model_three_region_overlap = True
model_three_score_threshold = 0.5
model_three_classifier = "multires"

#Model four settings:
model_four_activate = True
model_four_zoom = 1.0
model_four_region_overlap = True
model_four_score_threshold = 0.06
model_four_classifier = "multires"

#Model four settings:
model_five_activate = True
model_five_zoom = 0.6
model_five_region_overlap = False
model_five_score_threshold = 0.06
model_five_classifier = "multires"

#

# xView classes:
small =  [17,18,19,20,21,23,24,26,27,28,32,41,60,62,63,64,65,66,91]
medium = [11,12,15,25,29,33,34,35,36,37,38,42,44,47,50,53,56,59,61,71,72,73,76,84,86,93,94]
large  = [13,40,45,49,51,52,54,55,57,74,77,79,83,89]
common = [13,17,18,19,20,21,23,24,25,26,27,28,34,35,41,47,60,63,64,71,72,73,76,77,79,83,86,89,91]
rare   = [11,12,15,29,32,33,36,37,38,40,42,44,45,49,50,51,52,53,54,55,56,57,59,61,62,65,66,74,84,93,94]
