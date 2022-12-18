import random
import sys
try:
    from utils import (make_matching_plot_fast, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")
import numpy
import numpy as np
import numpy.linalg as linalg
import cv2 as cv
import math
import argparse
import matplotlib.cm  as cm

from  DeepLearning import get_dl_correspondences

def get_sift_correspondences(img1, img2,outlier):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    # sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()  # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    if outlier != None:
        outlier.reverse()
        for i in outlier:
            good_matches.pop(i-1)

    #good_matches = [good_matches[5],good_matches[14]]

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])





    #img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imshow('match', img_draw_match)
    #cv.imwrite('outlier.png',img_draw_match)
    #cv.waitKey(0)
    return points1, points2


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


def ransac(points1, points2, sample, pair):
    maxcount = -1

    if sample > 100:
        sample = 100

    dividend = 1
    divisor = 1
    for i in range(pair):
        divisor = divisor * (i + 1)
        dividend = dividend * (sample - i)
    comb = dividend // divisor

    times = 3000
    if comb < 3000:
        times = comb
    his = []

    for time in range(times):
        count = 0
        while True:
            randomlist = random.sample(range(sample), pair)
            randomlist.sort()
            if randomlist in his:
                continue
            else:
                his.append(randomlist)
                break
        samplepoints1 = []
        samplepoints2 = []
        for index in randomlist:
            samplepoints1.append(points1[index])
            samplepoints2.append(points2[index])

        samplepoints1 = np.array(samplepoints1)
        samplepoints2 = np.array(samplepoints2)

        homography = computehomography(samplepoints1, samplepoints2, pair)

        for index, point in enumerate(points1[:sample]):
            point2 = points2[index]
            point = numpy.append(point, 1)
            homocorres = np.matmul(homography, point)
            homocorres = homocorres / homocorres[2]
            if np.sqrt(np.sum(np.power(np.absolute(point2 - homocorres[:-1]), 2))) < 2:
                count = count + 1

        if count > maxcount:
            maxcount = count
            goodpoints1 = samplepoints1
            goodpoints2 = samplepoints2

        # print(count,maxcount)
    return goodpoints1, goodpoints2



def plotmatch(image0, image1, mkpts0,
                                mkpts1,path):
    H0, W0,channel0 = image0.shape
    H1, W1,channel1 = image1.shape
    H, W = max(H0, H1), W0 + W1

    out = 0 * np.ones((H, W,3), np.uint8)
    out[:H0, :W0,:channel0] = image0
    out[:H1, W0 :,:channel1] = image1
    #out = np.stack([out] * 3, -1)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)

    for (x0, y0), (x1, y1) in zip(mkpts0, mkpts1):
        color = list((np.random.choice(range(256),size=3)))
        c = [int(color[0]),int(color[1]),int(color[2])]
        cv.line(out, (x0, y0), (x1  + W0, y1),
                 color=c, thickness=1, lineType=cv.LINE_AA)
        # display line end-points as circles
        cv.circle(out, (x0, y0), 2, c, -1, lineType=cv.LINE_AA)
        cv.circle(out, (x1 +  W0, y1), 2, c, -1,
                   lineType=cv.LINE_AA)
    cv.imwrite(path,out)


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
        # print(np.argmin(s))
    else:
        solution = vh[8]

    homography = solution.reshape((3, 3))
    homography = homography * (1 / homography[2][2])
    return homography


def computenormhomography(normpoints1, normpoints2, pair, inversetransforms2, transforms1):
    homography = computehomography(normpoints1, normpoints2, pair)
    normhomography = np.matmul(inversetransforms2, np.matmul(homography, transforms1))
    normhomography = normhomography * (1 / normhomography[2][2])
    return normhomography


def computeerror(gt_correspondences, homography):
    error = 0.0
    for i in range(gt_correspondences.shape[1]):
        point1 = gt_correspondences[0][i]
        point1 = numpy.append(point1, 1)
        gt = gt_correspondences[1][i]
        homocorres = np.matmul(homography, point1)
        homocorres = homocorres / homocorres[2]
        error = error + np.sqrt(np.sum(np.power(gt - homocorres[:-1], 2)))
        #error = error + np.linalg.norm(gt - homocorres[:-1])
    error = error/gt_correspondences.shape[1]
    return error


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculating homography')
    parser.add_argument('--img1', type=str, help='img1 location', required=True)
    parser.add_argument('--img2', type=str, help='img2 location', required=True)
    parser.add_argument('--correspondence', type=str, help='correspondence file location', required=True)
    parser.add_argument('--ransac', action="store_true", help='ransac operation')
    parser.add_argument('--dl', action="store_true", help='deep learning operation')
    parser.add_argument('--pair', type=int, default=4, help='number of pair')
    parser.add_argument('--outlier', type=int, nargs='+')

    args = parser.parse_args()
    pair = args.pair
    outlier = args.outlier
    dloperation = args.dl
    ransacoperation = args.ransac



    gt_correspondences = np.load(args.correspondence)
    mean = 0.0

    img1 = cv.imread(args.img1)
    img2 = cv.imread(args.img2)

    grayimg1 = cv.imread(args.img1,cv.IMREAD_GRAYSCALE)
    grayimg2 = cv.imread(args.img2,cv.IMREAD_GRAYSCALE)

    if dloperation == True:
        points1, points2 = get_dl_correspondences(grayimg1, grayimg2,outlier)
    else:
        img1 = cv.imread(args.img1)
        img2 = cv.imread(args.img2)
        points1, points2 = get_sift_correspondences(img1, img2,outlier)

    plotmatch(img1, img2, points1, points2, './result/allpairs.png')

    if ransacoperation:
        points1, points2 = ransac(points1, points2, len(points1), pair)
    else:
        points1 = points1[:pair]
        points2 = points2[:pair]



    plotmatch(img1, img2, points1, points2,'./result/pairsforhomography.png')
    transforms1, transforms2, inversetransforms1, inversetransforms2 = normalize(points1, points2)

    normpoints1, normpoints2 = transform(points1, points2, transforms1, transforms2)

    homography = computehomography(points1, points2, pair)
    normhomography = computenormhomography(normpoints1, normpoints2, pair, inversetransforms2, transforms1)


    error = computeerror(gt_correspondences, homography)
    print("error of using point homography is {:.3f}".format(error))


    error = computeerror(gt_correspondences, normhomography)
    print("error of adding normalize operation is {:.3f}".format(error))






