import os
import math
import random
import torch
import torchvision.transforms as transforms
from torch.utils import data
import glob as gb
import numpy as np
import cv2
import csv
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
from PIL import Image
import warnings
import locality_aware_nms as nms_locality
#from geo_map_cython_lib import gen_geo_map

def computer_iou(box1, box2):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)
    im1 = np.zeros((512,512,1), dtype = "uint8")
    im2 = np.zeros((512,512,1), dtype = 'uint8')
    mask1 = cv2.fillPoly(im1, box1.reshape(1,-1, 2).astype(np.int32), 1)
    mask2 = cv2.fillPoly(im2, box2.reshape(1,-1, 2).astype(np.int32), 1)

    mask_and = cv2.bitwise_and(mask1, mask2)
    mask_or = cv2.bitwise_or(mask1, mask2)

    or_area = np.sum(np.float32(np.greater(mask_or,0)))
    and_area = np.sum(np.float32(np.greater(mask_and,0)))
    IOU = and_area/or_area
    return IOU




def get_proposals(score_map, geo_map, coord_ids, score_map_thresh=1e-10, box_thresh=1e-3, nms_thres=0.1):
    score = score_map.permute(0, 2, 3, 1)
    geometry = geo_map.permute(0, 2, 3, 1)
    score = score.data.cpu().numpy()
    geometry = geometry.data.cpu().numpy()
    boxes_ = []
    for i in range(score.shape[0]):
        score_map = score[i].squeeze()
        geo_map = geometry[i].squeeze()
        xy_text = np.argwhere(score_map > score_map_thresh)
        xy_text = xy_text[np.argsort(xy_text[:, 0])]

        text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :])
        #print('text_box_restored.shape', text_box_restored.shape)
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
        
        boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(score_map, mask)[0]

        if len(boxes) < 10:
            box = np.zeros((1,9), dtype=np.float32)
            boxs = np.tile(box, (10-len(boxes), 1))
            if len(boxes) == 0:
                boxes = boxs
            else:
                boxes = np.concatenate((boxes, boxs), axis=0)
        box_ind = boxes[:, 8].argsort()[-10:][::-1]
        boxes = boxes[box_ind]
        #print('boxes.shape', boxes.shape)
        boxes_.append(boxes)
        #print('boxes_.shape', len(boxes_))
    #print('coord_ids', coord_ids) 
    for i in range(score.shape[0]):
        box = boxes_[i]
        coord_id = coord_ids[i]
        #print('coord_id', coord_id)
        if len(coord_id) == 0:
            box[:, 8] = 0
        else:
            for j in range(len(box)):
                for k in range(len(coord_id)):
                    '''
                    print('box[0][:8]', box[j][:8])
                    print('coord_id[k][:8]', coord_id[k][:8])
                    raise RuntimeError
                    '''
                    iou = computer_iou(box[j][:8], coord_id[k][:8])
                    if iou > 0.5:
                        box[j][8] = coord_id[k][8]
                    else:
                        box[j][8] = 0
        boxes_[i] = box
    sm_masks = np.zeros((15,10,10))
    for i in range(len(boxes_)-1):
        box_id1 = boxes_[i][:,8]
        box_id2 = boxes_[i+1][:,8]
        for id1 in enumerate(box_id1):
            for id2 in enumerate(box_id2):
                if id1[1] == id2[1] and id1[1] != 0:
                    sm_masks[i, id1[0],id2[0]] = 1


    return np.asarray(boxes_), sm_masks


class sampler_for_video_clip(data.Sampler):
    def __init__(self, video_length):
        #self.batch_size = batch_size
        self.video_length = video_length
        self.pointer = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.pointer = self.pointer + 1
        return self.pointer - 1
    def __len__(self):
        return self.video_length
                    




def sort_order_for_video(name_list):
    '''
    sort the image&&txt'name in the right order 
    return:
    sorted_list -- a list of xxx name in the right order
    '''
    name_list = np.asarray(name_list)
    frame_num = [int(p.split('.')[0][6:]) for p in name_list]
    #print('frame_num', frame_num)
    index_num = list(np.argsort(frame_num))
    #print('index_num', index_num)
    sorted_list = name_list[index_num]
    return sorted_list


def load_annoataion(p):
    '''
    load annotation from the text file

    Note:
    modified
    1. top left vertice
    2. clockwise

    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    coord_ids = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader: 
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            quality_label = line[-1]# strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            text_id = line[-2]              
            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(int, line[:8]))  
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            coord_ids.append(list(map(int, line[:9])))
            if quality_label == 'LOW':
                text_tags.append(True)
            else:
                text_tags.append(False)
        
    return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool), coord_ids
    #return text_polys, text_tags

def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    poly_ = np.array(poly)
    assert poly_.shape == (4,2), 'poly shape should be 4,2'
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.

def calculate_distance(c1, c2):
    return math.sqrt(math.pow(c1[0]-c2[0], 2) + math.pow(c1[1]-c2[1], 2))

def choose_best_begin_point(pre_result):
    """
    find top-left vertice and resort
    """
    final_result = []
    for coordinate in pre_result:
        x1 = coordinate[0][0]
        y1 = coordinate[0][1]
        x2 = coordinate[1][0]
        y2 = coordinate[1][1]
        x3 = coordinate[2][0]
        y3 = coordinate[2][1]
        x4 = coordinate[3][0]
        y4 = coordinate[3][1]
        xmin = min(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        xmax = max(x1, x2, x3, x4)
        ymax = max(y1, y2, y3, y4)
        combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                     [[x2, y2], [x3, y3], [x4, y4], [x1, y1]], 
                     [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], 
                     [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
        dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
        force = 100000000.0
        force_flag = 0
        for i in range(4):
            temp_force = calculate_distance(combinate[i][0], dst_coordinate[0]) + calculate_distance(combinate[i][1], dst_coordinate[1]) + calculate_distance(combinate[i][2], dst_coordinate[2]) + calculate_distance(combinate[i][3], dst_coordinate[3])
            if temp_force < force:
                force = temp_force
                force_flag = i
        #if force_flag != 0:
        #    print("choose one direction!")
        final_result.append(combinate[force_flag])
        
    return final_result


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''

    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys, tags
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []

    # find top-left and clockwise
    polys = choose_best_begin_point(polys)

    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            #print('invalid poly')
            continue
        if p_area > 0:
            #print('poly in wrong direction')
            poly = np.asarray(poly)[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)

def crop_area(im, polys, tags, coord_ids, crop_background=False, max_tries=5000, vis = False, img_name = None):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    
    if polys.shape[0] == 0:
        return im, [], [], []

    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags, coord_ids
    
    for i in range(max_tries):
        #print('we have try {} times'.format(i))
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        # if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
        if xmax - xmin < 0.1*w or ymax - ymin < 0.1*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            # no text in this area
            if crop_background == True:
                im = im[ymin:ymax+1, xmin:xmax+1, :]
                polys = []
                tags = []

                return im, polys, tags, coord_ids
            else:
                continue
        else:
            if crop_background == False:
                im = im[ymin:ymax+1, xmin:xmax+1, :]
                polys = polys.tolist()
                polys = [polys[i] for i in selected_polys]
                polys = np.array(polys)
                polys[:, :, 0] -= xmin #ndarray
                polys[:, :, 1] -= ymin

                polys = polys.astype(np.int32)
                polys = polys.tolist()

                tags  = tags.tolist()
                tags  = [tags[i]  for i in selected_polys]
                coord_ids = [coord_ids[i] for i in selected_polys]
                #print('coord_ids in crop area', coord_ids)
                #print('coord_ids[0][2]', coord_ids[0][2])
                for i in range(len(coord_ids)):
                    coord_ids[i][0] -= xmin
                    coord_ids[i][2] -= xmin
                    coord_ids[i][4] -= xmin
                    coord_ids[i][6] -= xmin
                    coord_ids[i][1] -= ymin
                    coord_ids[i][3] -= ymin
                    coord_ids[i][5] -= ymin
                    coord_ids[i][7] -= ymin
                return im, polys, tags, coord_ids
            else:
                continue
    return im, polys, tags, coord_ids



"""
def crop_area(im, polys, tags, crop_background=False, max_tries=50, vis = True, img_name = None):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    print('goggogoogo')
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    print('aaaaaaaaa')
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    
    print('bbbbbbbbb')
    for i in range(max_tries):
        print('we have try {} times'.format(i))
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        # if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
        if xmax - xmin < 0.1*w or ymax - ymin < 0.1*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                im = im[ymin:ymax+1, xmin:xmax+1, :]
                polys = polys[selected_polys]
                tags = tags[selected_polys]
                if vis == True:
                    path = os.path.join(os.path.abspath('./'), 'tmp/vis_for_crop', '{}-bg.jpg'.format(img_name))
                    cv2.imwrite(path, im)
                    print('save a bg')
                return im, polys, tags
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin

        print('crop front')
        if vis == True:
            #print('TEST for visualization about crop img')
            for ids, poly in enumerate(polys):
                print('img h:{} w:{} poly id:{} {}'.format(im.shape[0], im.shape[1], ids, poly))
                x = [poly.astype(np.int32).reshape((-1, 1, 2))]
                cv2.polylines(im[:, :, ::-1], x, True, color=(255, 255, 0), thickness=3)
                print(x)
            path = os.path.join(os.path.abspath('./'), 'tmp/vis_for_crop', '{}-fg.jpg'.format(img_name))
            cv2.imwrite(path, im)
            print('save a fg')
        return im, polys, tags

    return im, polys, tags
"""

def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    distance = 0
    try:
        eps = 1e-5
        distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) /(np.linalg.norm(p2 - p1)+eps)
    
    except:
        print('point dist to line raise Exception')
    
    return distance


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        '''
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        '''
        if angle/np.pi * 180 > 45:
            
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def generate_rbox(im_size, polys, tags):
    """
    score map is (128, 128, 1) with shrinked poly
    poly mask is (128, 128, 1) with differnt colors


    geo map is  (128, 128, 5) with
    """
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    for poly_idx, poly_tag in enumerate(zip(polys, tags)):
        poly = poly_tag[0]
        tag = poly_tag[1]
        poly = np.array(poly)
        tag  = np.array(tag)
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)

        # use different color to draw poly mask
        cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
        # if the poly is too small, then ignore it during training
        poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
        poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
        # if min(poly_h, poly_w) < FLAGS.min_text_size:
        if min(poly_h, poly_w) < 10:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
        if tag:
            cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]

            #fit_line ([x1, x2], [y1, y2]) return k, -1, b just a line
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])             #p0, p1
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])    #p0, p3
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])     #p1, p2

            #select shorter line
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                
                if edge[1] == 0:#verticle
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
               
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])

        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[[min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)
        #print('parallel {} rectangle {}'.format(parallelogram, rectange))
        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        # this is one area of many
        
        for y, x in xy_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle
        
        #gen_geo_map.gen_geo_map(geo_map, xy_in_poly, rectange, rotate_angle)

    ###sum up
    # score_map , in shrinked poly is 1
    # geo_map, corresponding to score map
    # training map is less than geo_map

    return score_map, geo_map, training_mask

def image_label(txt_root, 
                image_list, img_name,
                txt_list, txt_name,
                index,
                input_size = 512, 
                random_scale = 1.0,
                background_ratio = 3./8):
    '''
    get image's corresponding matrix and ground truth
    return
    images [512, 512, 3]
    score  [128, 128, 1]
    geo    [128, 128, 5]
    mask   [128, 128, 1]
    '''

    try:
        im_fn = image_list[index]
        #print('index', index)
        txt_fn = txt_list[index]
        #print('im_fn:{} txt_fn:{}'.format(im_fn, txt_fn))
        im = cv2.imread(im_fn)#h, w, (BGR)
        # print im_fn
        h, w, _ = im.shape
        #txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
        if not os.path.exists(txt_fn):
            sys.exit('text file {} does not exists'.format(txt_fn))

        text_polys, text_tags, coord_ids = load_annoataion(txt_fn)
        #print('text_polys', text_polys)
        #print('text_tags', text_tags)
        #print('coord_ids', coord_ids)

        #print('text_polys', text_polys)
        #print('text_tags', text_tags)
        text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
        if len(text_polys) == 0:
            score_map = np.zeros((input_size, input_size), dtype=np.uint8)
            geo_map_channels = 5
            geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
            training_mask = np.ones((input_size, input_size), dtype=np.uint8)
            images = im[:, :, ::-1].astype(np.float32)
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)
            coord_ids = []
            images = cv2.resize(images, dsize=(512, 512)) 
            return images, score_maps, geo_maps, training_masks, coord_ids
        '''
        if text_polys.shape[0] == 0:
             continue
        '''
        # random scale this image
        #rd_scale = random_scale

        im = cv2.resize(im, dsize=None, fx=random_scale, fy=random_scale)
        text_polys *= random_scale
        for i in range(len(coord_ids)):
            for j in range(8):
                coord_ids[i][j] *=random_scale
        ###########################for exception to return #############################
        h, w, _ = im.shape

        # pad the image to the training input size or the longer side of image
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = im_padded
        # resize the image to input size
        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        im = cv2.resize(im, dsize=(resize_w, resize_h))
        resize_ratio_1_x = resize_w/float(new_w)
        resize_ratio_1_y = resize_h/float(new_h)
        #print(text_polys.type.name)
        #print('resize_ratio_1_x', resize_ratio_1_x)
        for i in range(len(text_polys)):
            for j in range(4):
                #print('before text_polys[{}][{}][0]:{}'.format(i,j,text_polys[i][j][0]))
                text_polys[i][j][0] *= resize_ratio_1_x
                #print('text_polys[{}][{}][0]:{}'.format(i, j, text_polys[i][j][0]))
                text_polys[i][j][1] *= resize_ratio_1_y
        #print('len(coord_ids)', len(coord_ids))
        for i in range(len(coord_ids)):
            coord_ids[i][0] *= resize_ratio_1_x
            #print('coord_ids[{}][0]:{}'.format(i, coord_ids[i][0]))
            coord_ids[i][1] *= resize_ratio_1_y
            coord_ids[i][2] *= resize_ratio_1_x
            coord_ids[i][3] *= resize_ratio_1_y
            coord_ids[i][4] *= resize_ratio_1_x
            coord_ids[i][5] *= resize_ratio_1_y
            coord_ids[i][6] *= resize_ratio_1_x
            coord_ids[i][7] *= resize_ratio_1_y
        #text_polys[:, :, 0] *= resize_ratio_3_x
        #ext_polys[:, :, 1] *= resize_ratio_3_y
        #print('text_ploys after resize 1', text_polys)
        #print('coord_ids after resize 1', coord_ids)
        new_h, new_w, _ = im.shape

        im, text_polys, text_tags, coord_ids = crop_area(im, text_polys, text_tags, coord_ids, crop_background=False)
        count = 0
        for i,tags in enumerate(text_tags):
            if  tags:
                del coord_ids[i-count]
                count += 1
        h, w, _ = im.shape

        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = im_padded
        # resize the image to input size
        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        im = cv2.resize(im, dsize=(resize_w, resize_h))
        resize_ratio_2_x = resize_w/float(new_w)
        resize_ratio_2_y = resize_h/float(new_h)
        #print('text_polys before resize 2', text_polys)
        #print('text_polys before resize 2', coord_ids)
        for i in range(len(text_polys)):
            for j in range(4):
                text_polys[i][j][0] *= resize_ratio_2_x
                text_polys[i][j][1] *= resize_ratio_2_y
        for i in range(len(coord_ids)):
            coord_ids[i][0] *= resize_ratio_2_x
            coord_ids[i][1] *= resize_ratio_2_y
            coord_ids[i][2] *= resize_ratio_2_x
            coord_ids[i][3] *= resize_ratio_2_y
            coord_ids[i][4] *= resize_ratio_2_x
            coord_ids[i][5] *= resize_ratio_2_y
            coord_ids[i][6] *= resize_ratio_2_x
            coord_ids[i][7] *= resize_ratio_2_y
        new_h, new_w, _ = im.shape
        #print('new_h{}new_w{}'.format(new_h, new_w))
        #print('text_polys after select', text_polys)
        #print('coord_ids after select', coord_ids)
        score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

    except Exception as e:
        print(e)
        #raise RuntimeError
        print('Exception continue')
        return None, None, None, None, None

    images = im[:, :, ::-1].astype(np.float32)
    score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
    geo_maps = geo_map[::4, ::4, :].astype(np.float32)
    training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

    return images, score_maps, geo_maps, training_masks, coord_ids
    '''
        ########################################################################

        # print rd_scale
        # random crop a area from image
        #if np.random.rand() < background_ratio:
        #tmp = False
        # we do not crop background as input should be a video clip
    
        if np.random.rand() < background_ratio:
            # crop background
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
            assert len(text_polys) == 0, 'crop area should have no text_polys'
            #if text_polys.shape[0] > 0:
            #    print('cannot find background')
            #    return None, None, None, None 
            
            # pad and resize image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = cv2.resize(im_padded, dsize=(input_size, input_size))
            score_map = np.zeros((input_size, input_size), dtype=np.uint8)
            geo_map_channels = 5 
            geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
            training_mask = np.ones((input_size, input_size), dtype=np.uint8)
        else:
            im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
            #assert len(text_polys) > 0, 'crop area should have some text_polys'
            if len(text_polys) == 0: #for some reason , gt contain no polys, have to return black
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map_channels = 5 
                geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                images = im[:, :, ::-1].astype(np.float32)
                score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
                geo_maps = geo_map[::4, ::4, :].astype(np.float32)
                training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)
                return images, score_maps, geo_maps, training_masks
            #if text_polys.shape[0] == 0:
            #    print('cannot find frontground')
            #    return None, None, None, None
            h, w, _ = im.shape

            # pad the image to the training input size or the longer side of image
            new_h, new_w, _ = im.shape
            max_h_w_i = np.max([new_h, new_w, input_size])
            im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
            im_padded[:new_h, :new_w, :] = im.copy()
            im = im_padded
            # resize the image to input size
            new_h, new_w, _ = im.shape
            resize_h = input_size
            resize_w = input_size
            im = cv2.resize(im, dsize=(resize_w, resize_h))
            resize_ratio_3_x = resize_w/float(new_w)
            resize_ratio_3_y = resize_h/float(new_h)
            #print(text_polys.type.name)

            for i in range(len(text_polys)):
                for j in range(4):
                    text_polys[i][j][0] *= resize_ratio_3_x
                    text_polys[i][j][1] *= resize_ratio_3_y

            #text_polys[:, :, 0] *= resize_ratio_3_x
            #ext_polys[:, :, 1] *= resize_ratio_3_y
            new_h, new_w, _ = im.shape
            #print('done3')
            score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
            #print('done4')
        
    except Exception as e:
        raise RuntimeError
        print('Exception continue')
        return None, None,None,None

    images = im[:, :, ::-1].astype(np.float32)
    score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
    geo_maps = geo_map[::4, ::4, :].astype(np.float32)
    training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

    return images, score_maps, geo_maps, training_masks
    '''
 
def transform_for_train(img):
    """
    args 
    img -- 
    """
    h, w, c = img.shape
    assert h == 512, 'img should be 512'
    assert w == 512, 'img should be 512'
    assert c == 3  , 'img should be 3 channels'
    # cv2 trans to pil
    image = Image.fromarray(np.uint8(img))

    transform_list = []
    
    transform_list.append(transforms.ColorJitter(0.5, 0.5, 0.5, 0.25))

    transform_list.append(transforms.ToTensor())
    
    transform_list.append(transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)))

    transform = transforms.Compose(transform_list)

    transforms.Compose(transform_list)
    return transform(image)

class custom_dset(data.Dataset):
    def __init__(self, video_name_path):
        #print('video_name_path', video_name_path)
        self.random_scale = np.random.choice(np.array([0.5, 1.0, 2.0, 3.0]))
        frame_path = os.path.join(video_name_path,'frame/')
        #print('frame_path', frame_path)
        txt_path = os.path.join(video_name_path, 'gt/')
        frame_list = sorted([p for p in os.listdir(frame_path) if p.split('.')[1] == 'jpg'])
        txt_list = sorted([p for p in os.listdir(txt_path) if p.split('.')[1] == 'txt'])
        #print('frame_list', frame_list)
        #print('gt_list', gt_list)
        assert len(frame_list) == len(txt_list), 'in {} frame length do not match label length'.format(video_name_path)
        #raise RuntimeError
        self.sorted_frame_list = sort_order_for_video(frame_list)  
        self.sorted_txt_list = sort_order_for_video(txt_list)
        #print('sorted_frame_list', self.sorted_frame_list)
        #print('sorted_txt_list', self.sorted_txt_list)
        
        self.video_name_path = video_name_path
        self.txt_path = txt_path
        self.img_path_list = [os.path.join(frame_path,p) for p in self.sorted_frame_list]
        self.txt_path_list = [os.path.join(txt_path, p) for p in self.sorted_txt_list]
        #print('img_path_list', self.img_path_list)
        #print('txt_path_list', self.txt_path_list)
        
        '''
        # check img_path_list, img_name_list, txt_root
        for i in range(len(self.img_path_list)):
            img_id = []
            img_id.append(os.path.basename(self.img_path_list[i]).strip('.jpg'))
            img_id.append(os.path.basename(self.txt_path_list[i]).strip('.txt'))
            img_id.append(self.img_name_list[i].strip('.jpg'))
            img_id.append(self.txt_name_list[i].strip('.txt'))
            if (img_id[0] == img_id[1])&(img_id[2] == img_id[3])&(img_id[0] == img_id[2]):
                continue
            else:
                print(img_id[0])
                print(img_id[1])
                print(img_id[2])
                print(img_id[3])
                sys.exit('img list and txt list is not matched')
        '''

    def __getitem__(self, index):
        #transform = transform_for_train()
        status = True
        while status:
            img, score_map, geo_map, training_mask, coord_ids = image_label(self.txt_path,
            
                self.img_path_list, self.sorted_frame_list,
            
                self.txt_path_list, self.sorted_txt_list,
            
                index, input_size = 512,
            
                random_scale = self.random_scale, background_ratio = 3./8)
        
            if not img is None:#512,512,3 ndarray should transform to 3,512,512


                status = False
                
                #img = transform_for_train(img)
                img = img.transpose(2, 0, 1)
                #print(img.shape)
                #print(type(img))

                return img, score_map, geo_map, training_mask, coord_ids

            else:

                #index = np.random.random_integers(index-8, index+8)
                index = index - 1
                print('Exception in getitem, and choose another index:{}'.format(index))



        
        #    sys.exit('some image cant find approprite crop')
        #img = transform_for_train(img)
        #if img == None:
        #   return None, None, None, None

        

    def __len__(self):
        return len(self.img_path_list)

def collate_fn(batch):
    img, score_map, geo_map, training_mask, coord_ids = zip(*batch)#tuple
    '''
    for i in range(len(img)):
        print('img.shape', img[i].shape)
    '''
    bs = len(score_map)
    images = []
    score_maps = []
    geo_maps = []
    training_masks = []
    for i in range(bs):
        if img[i] is not None:
            a = torch.from_numpy(img[i])
            #a = img[i]
            images.append(a)
           
            b = torch.from_numpy(score_map[i])
            b = b.permute(2, 0, 1)
            score_maps.append(b)
            
            c = torch.from_numpy(geo_map[i])
            c = c.permute(2, 0, 1)
            geo_maps.append(c)
            
            d = torch.from_numpy(training_mask[i])
            d = d.permute(2, 0, 1)
            training_masks.append(d)
    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

    return images, score_maps, geo_maps, training_masks, coord_ids
## img = bs * 512 * 512 *3
## score_map = bs* 128 * 128 * 1
## geo_map = bs * 128 * 128 * 5
## training_mask = bs * 128 * 128 * 1
