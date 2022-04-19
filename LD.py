#  -*-  coding:utf-8  -*- 
#  ------------------------------
#  Author: Kangdong Jin
#    Date: 2021/12/15 13:13
#     For: 
#  ------------------------------
import cv2
import numpy as np


class LaneDetection:
    # some of the codes is functional with other project, if there are some misunderstanding, just ignore


    def LD(picture = None):


        # lines : np.array([])
        # lines_Hough_al : np.array([])

        left_lines_fitted, right_lines_fitted = np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]])

        # read the orignal picture
        if picture is not None:
            # gray scale + Canny
            frame = LaneDetection.app_canny(picture)
            # make mask
            mask = LaneDetection.mask(frame)
            # put mask on the cannyed grayscale frame
            masked_edge_img = cv2.bitwise_and(frame, mask)
            # display the cannyed + masked grayscale frame
            # LaneDetection.show_masked(canny_remote, masked_edge_img)

            # do Hough Transform on the cannyed + masked grayscale frame, and
            # seperate the lines into left ones and right ones
            lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 100, 15, minLineLength=00, maxLineGap=20)

            if lines is not None:
                left_lines = [line for line in lines if -0.5 > LaneDetection.calculate_slope(line)]
                right_lines = [line for line in lines if LaneDetection.calculate_slope(line) > 0.5]

                # Remove noisy lines
                left_lines = LaneDetection.reject_abnormal_lines(left_lines, 0.1)
                right_lines = LaneDetection.reject_abnormal_lines(right_lines, 0.1)

                # Fit the left and right lane lines separately
                left_lines_fitted = LaneDetection.least_squares_fit(left_lines)
                right_lines_fitted = LaneDetection.least_squares_fit(right_lines)

                if left_lines_fitted is None:
                    left_lines_fitted = np.array([[0, 0], [0, 0]])
                if right_lines_fitted is None:
                    right_lines_fitted = np.array([[0, 0], [0, 0]])

            return left_lines_fitted, right_lines_fitted

        return left_lines_fitted, right_lines_fitted


    # slope_calculation
    def calculate_slope(line):
        x_1, y_1, x_2, y_2 = line[0]             # line:[[x1,y1,x2,y2], [], []...]
        return (y_2 - y_1) / (x_2 - x_1 + 0.01)  # calculate the scope of every single line


    # Package of Canny
    def app_canny(picture):

        img = picture
        minThreshold = 60
        maxThreshold = 130
        edges = cv2.Canny(img, minThreshold, maxThreshold)

        return edges

    # mask making(trapezoid area in front of the car)
    def mask(frame):
        mask = np.zeros_like(frame)

        height = mask.shape[0]
        height_bevel = int(0.75 * height)
        length = mask.shape[1]
        length_bevel = int(0.6 * length)
        length_under = int(0.2 * length)
        mask = cv2.fillPoly(mask, np.array([[[length_under, height], [length - length_bevel, height_bevel],
                                             [length_bevel, height_bevel], [length - length_under, height],
                                             [length_under, height]]]),
                            color=255)

        ########## the size of mask and the parameters in cv2.fillPoly() needs to be modified
        ########## for different scenarios (camera pointing angle and orientation), in order
        ########## to achieve the effect of recognition of the specified area in the image

        return mask


    # firstly calculate the mean slope in both lists of lines, if there are abnormal lines, which have
    # great deviation with others in the same side, they will be rejected as noise. After rejection will
    # the lines be fitted into left and right lanes by using least square fitting.
    def reject_abnormal_lines(lines, threshold):
        slopes = [LaneDetection.calculate_slope(line) for line in lines]
# -----------------------------------------------------------------------------------------------------
        # i = 0
        # while True:
        #     if i + 1 > len(slopes):                               # these codes equals to the parameter
        #         break                                             # tuning in line 69 / 70
        #     if 0.5 > abs(slopes[i]):
        #         slopes.pop(i)
        #         lines.pop(i)
        #         i = i
        #     else:
        #         i += 1
# -----------------------------------------------------------------------------------------------------
        while len(slopes) > 0:
            mean = np.mean(slopes)
            diff = [abs(slope - mean) for slope in slopes]
            max_slope = np.argmax(diff)
            if diff[max_slope] > threshold:
                slopes.pop(max_slope)
                lines.pop(max_slope)
            else:
                break
        return lines


    # least square fitting
    def least_squares_fit(lines):
        """
        :param (line in lines): set of lines， [np.array([x_1, y_1, x_2, y_2]), [np.array([x_1, y_1, x_2, y_2]),...]]
        :return: end points of a line， np.array([[xmin, ymin], [xmax, ymax]])
        """
        # 1.取出所有坐标点
        x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines]) # 第一个[0]代表横向，第二位[]代表取位
        y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
        # 1.进行直线拟合，得到多项式系数
        if lines != None:
            if len(x_coords) >= 1:
                poly = np.polyfit(x_coords, y_coords, deg=1)
                # 1.根据多项式系数，计算两个直线上的点，用于唯一确定这条直线
                point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
                point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
                return np.array([point_min, point_max], dtype=np.int)
            else:
                pass
        else:
            pass


## error report
class CustomError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self)
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo
