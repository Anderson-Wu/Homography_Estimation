import random
import sys

import numpy
import numpy as np
import numpy.linalg as linalg
import cv2 as cv
import math
import argparse

WINDOW_NAME = 'window'

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])


def normalize(points1, points2):
    means1coord = np.mean(points1, axis=0)
    means2coord = np.mean(points2, axis=0)
    means1coordx, means1coordy = means1coord[0], means1coord[1]
    means2coordx, means2coordy = means2coord[0], means2coord[1]

    shifts1 = points1 - [means1coordx, means1coordy]
    shifts2 = points2 - [means2coordx, means2coordy]

    meanpoints1dis = np.mean(np.sqrt(np.sum(np.power(shifts1, 2), axis=1)))
    meanpoints2dis = np.mean(np.sqrt(np.sum(np.power(shifts2, 2), axis=1)))

    scales1 = math.sqrt(2) / meanpoints1dis
    scales2 = math.sqrt(2) / meanpoints2dis

    scalemats1 = np.array([[scales1, 0, 0], [0, scales1, 0], [0, 0, 1]])
    scalemats2 = np.array([[scales2, 0, 0], [0, scales2, 0], [0, 0, 1]])

    translates1 = np.array([[1, 0, -means1coordx], [0, 1, -means1coordy], [0, 0, 1]])
    translates2 = np.array([[1, 0, -means2coordx], [0, 1, -means2coordy], [0, 0, 1]])

    transforms1 = np.matmul(scalemats1, translates1)
    transforms2 = np.matmul(scalemats2, translates2)

    inversetransforms1 = np.linalg.inv(transforms1)
    inversetransforms2 = np.linalg.inv(transforms2)

    return transforms1, transforms2, inversetransforms1, inversetransforms2


def transform(points1, points2, transforms1, transforms2):
    norms1 = None
    for vec in points1:
        homo = np.concatenate([vec, [1]])
        res = transforms1.dot(homo)
        if norms1 is None:
            norms1 = res
        else:
            norms1 = np.vstack([norms1, res])

    norms2 = None
    for vec in points2:
        homo = np.concatenate([vec, [1]])
        res = transforms2.dot(homo)
        if norms2 is None:
            norms2 = res
        else:
            norms2 = np.vstack([norms2, res])
    return norms1, norms2




def computehomography(points1, points2, pair):
    matrix = np.array([], dtype=np.float64)
    for i in range(pair):
        u1 = points1[i][0]
        v1 = points1[i][1]
        u2 = points2[i][0]
        v2 = points2[i][1]

        equ1 = np.array([u1, v1, 1, 0, 0, 0, -u2 * u1, -u2 * v1, -u2], dtype=np.float64)
        equ2 = np.array([0, 0, 0, u1, v1, 1, -v2 * u1, -v2 * v1, -v2], dtype=np.float64)
        matrix = np.concatenate([matrix, equ1])
        matrix = np.concatenate([matrix, equ2])

    matrix = matrix.reshape((pair * 2), 9)
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)

    if len(s) == 9:
        solution = vh[np.argmin(s)]
    else:
        solution = vh[8]

    homography = solution.reshape((3, 3))
    homography = homography * (1 / homography[2][2])
    return homography


def computenormhomography(normpoints1, normpoints2, pair,inversetransforms2,transforms1):
    homography = computehomography(normpoints1,normpoints2,pair)
    normhomography = np.matmul(inversetransforms2, np.matmul(homography, transforms1))

    return normhomography




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculating homography')
    parser.add_argument('--img',type=str,help='img location',required=True)
    args = parser.parse_args()




    img = cv.imread(args.img)


    points_add = []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break  # exist when pressing ESC

    cv.destroyAllWindows()
    cv.imwrite('choose_point.png', img_)
    #print('{} Points added'.format(len(points_add)))




    points2 = np.array(points_add)
    points1 = np.array([[0,0],[0,639],[479,639],[479,0]])
    pair = 4

    transforms1, transforms2, inversetransforms1, inversetransforms2 = normalize(points1, points2)
    normpoints1, normpoints2 = transform(points1, points2, transforms1, transforms2)

    homography = computehomography(points1, points2, pair)
    normhomography = computenormhomography(normpoints1, normpoints2, pair,inversetransforms2,transforms1)






    result = np.zeros((640,480,3),dtype=np.float64)
    imgh = img.shape[0]
    imgw = img.shape[1]
    for i in range(480):
        for j in range(640):
            point1 = numpy.append([i+0.5,j+0.5], 1)
            homocorres = np.matmul(homography, point1)
            homocorres = homocorres / homocorres[2]
            point2 = homocorres[:2]
            point2 = point2-0.5

            if point2[0] < 0 or point2[0] >= (imgw-1) or point2[1] < 0 or point2[1] >= (imgh-1):
                result[j][i] = np.array([0,0,0])
            else:
                leftup = np.array([math.floor(point2[0]),math.floor(point2[1])])
                rightup = np.array([math.ceil(point2[0]),math.floor(point2[1])])
                leftdown = np.array([math.floor(point2[0]), math.ceil(point2[1])])
                rightdown = np.array([math.ceil(point2[0]), math.ceil(point2[1])])

                valleftup = img[leftup[1]][leftup[0]]
                valrightup = img[rightup[1]][rightup[0]]

                valleftdown = img[leftdown[1]][leftdown[0]]
                valrightdown = img[rightdown[1]][rightdown[0]]

                arealeftup = abs(np.prod(point2-leftup))
                arearightup = abs(np.prod(point2 - rightup))
                arealeftdown = abs(np.prod(point2 - leftdown))
                arearightdown = abs(np.prod(point2 - rightdown))


                val = arealeftup*valrightdown + arearightup*valleftdown + arealeftdown*valrightup + arearightdown*valleftup
                if val[0] > 255:
                    val[0] = 255
                elif val[0] < 0:
                    val[0] = 0

                if val[1] > 255:
                    val[1] = 255
                elif val[1] < 0:
                    val[1] = 0

                if val[2] > 255:
                    val[2] = 255
                elif val[2] < 0:
                    val[2] = 0

                result[j][i] = np.around(val)


    cv.imshow('result',result.astype(np.uint8))
    cv.imwrite('./result/book.png', result.astype(np.uint8))
    cv.waitKey(0)




