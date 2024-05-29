import cv2
import numpy as np
import os
import shutil
import math

class BuildingFootprintRegularization:
    def __init__(self, input_folder):
        self.input_folder = input_folder

    def line(self,p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    def intersection(self,L1, L2):
        D  = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False

    def par_line_dist(self,L1, L2):
        A1, B1, C1 = L1
        A2, B2, C2 = L2
        new_C1 = C1 / A1
        new_A2 = 1
        new_B2 = B2 / A2
        new_C2 = C2 / A2

        dist = (np.abs(new_C1-new_C2))/(np.sqrt(new_A2*new_A2+new_B2*new_B2))
        return dist

    def point_in_line(self,m, n, x1, y1, x2, y2):
        x = (m * (x2 - x1) * (x2 - x1) + n * (y2 - y1) * (x2 - x1) + (x1 * y2 - x2 * y1) * (y2 - y1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        y = (m * (x2 - x1) * (y2 - y1) + n * (y2 - y1) * (y2 - y1) + (x2 * y1 - x1 * y2) * (x2 - x1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        return (x, y)

    def Nrotation_angle_get_coor_coordinates(self,point, center, angle):
        src_x, src_y = point
        center_x, center_y = center
        radian = math.radians(angle)

        dest_x = (src_x - center_x) * math.cos(radian) + (src_y - center_y) * math.sin(radian) + center_x
        dest_y = (src_y - center_y) * math.cos(radian) - (src_x - center_x) * math.sin(radian) + center_y

        return (dest_x, dest_y)

    def Srotation_angle_get_coor_coordinates(self,point, center, angle):
        src_x, src_y = point
        center_x, center_y = center
        radian = math.radians(angle)
        dest_x = (src_x - center_x) * math.cos(radian) - (src_y - center_y) * math.sin(radian) + center_x
        dest_y = (src_x - center_x) * math.sin(radian) + (src_y - center_y) * math.cos(radian) + center_y
        return (dest_x, dest_y)
    def cal_dist(self,point_1, point_2):
        dist = np.sqrt(np.sum(np.power((point_1-point_2), 2)))
        return dist
    def cal_ang(self,point_1, point_2, point_3):
        a = math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1] - point_3[1]))
        b = math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
        c = math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))
        B = math.degrees(math.acos((b*b-a*a-c*c)/(-2*a*c)))
        return B
    def azimuthAngle(self,point_0, point_1):
        x1, y1 = point_0
        x2, y2 = point_1
        if x1 < x2:
            if y1 < y2:
                ang = math.atan((y2 - y1) / (x2 - x1))
                ang = ang * 180 / math.pi
                return ang
            elif y1 > y2:
                ang = math.atan((y1 - y2) / (x2 - x1))
                ang = ang * 180 / math.pi
                return 90 + (90 - ang)
            elif y1 == y2:
                return 0
        elif x1 > x2:
            if y1 < y2:
                ang = math.atan((y2-y1)/(x1-x2))
                ang = ang*180/math.pi
                return 90+(90-ang)
            elif y1 > y2:
                ang = math.atan((y1-y2)/(x1-x2))
                ang = ang * 180 / math.pi
                return ang
            elif y1 == y2:
                return 0
        elif x1 == x2:
            return 90
    def pldist(self,x0, x1, x2):
        x0, x1, x2 = x0[:2], x1[:2], x2[:2]
        if x1[0] == x2[0]:
            return np.abs(x0[0] - x1[0])

        return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                        np.linalg.norm(x2 - x1))
    
    def _rdp(self,M, epsilon, dist):
        dmax = 0.0
        index = -1
        for i in range(1, M.shape[0]):
            d = dist(M[i], M[0], M[-1])
            if d > dmax:
                index = i
                dmax = d
        if dmax > epsilon:
            r1 = self._rdp(M[:index + 1], epsilon, dist)
            r2 = self._rdp(M[index:], epsilon, dist)
            return np.vstack((r1[:-1], r2))
        else:
            return np.vstack((M[0], M[-1]))
    def _rdp_nn(self,seq, epsilon, dist):
        return self._rdp(np.array(seq), epsilon, dist).tolist()
    def rdp(self,M, epsilon=0):
        dist=self.pldist
        if "numpy" in str(type(M)):
            return self._rdp(M, epsilon, dist)
        return self._rdp_nn(M, epsilon, dist)

    def boundary_regularization(self,img, epsilon=6):
        h, w = img.shape[0:2]
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = np.squeeze(contours[0])
        contours = self.rdp(contours, epsilon=epsilon)

        contours[:, 1] = h - contours[:, 1]
        dists = []
        azis = []
        azis_index = []
        for i in range(contours.shape[0]):
            cur_index = i
            next_index = i+1 if i < contours.shape[0]-1 else 0
            prev_index = i-1
            cur_point = contours[cur_index]
            nest_point = contours[next_index]
            dist = self.cal_dist(cur_point, nest_point)
            azi = self.azimuthAngle(cur_point, nest_point)
            dists.append(dist)
            azis.append(azi)
            azis_index.append([cur_index, next_index])
        longest_edge_idex = np.argmax(dists)
        main_direction = azis[longest_edge_idex]
        correct_points = []
        para_vetr_idxs = []
        for i, (azi, (point_0_index, point_1_index)) in enumerate(zip(azis, azis_index)):
            if i == longest_edge_idex:
                correct_points.append([contours[point_0_index], contours[point_1_index]])
                para_vetr_idxs.append(0)
            else:
                rotate_ang = main_direction - azi
                if np.abs(rotate_ang) < 180/4:
                    rotate_ang = rotate_ang
                    para_vetr_idxs.append(0)
                elif np.abs(rotate_ang) >= 90-180/4:
                    rotate_ang = rotate_ang + 90
                    para_vetr_idxs.append(1)
                point_0 = contours[point_0_index]
                point_1 = contours[point_1_index]
                point_middle = (point_0 + point_1) / 2
                if rotate_ang > 0:
                    rotate_point_0 = self.Srotation_angle_get_coor_coordinates(point_0, point_middle, np.abs(rotate_ang))
                    rotate_point_1 = self.Srotation_angle_get_coor_coordinates(point_1, point_middle, np.abs(rotate_ang))
                elif rotate_ang < 0:
                    rotate_point_0 = self.Nrotation_angle_get_coor_coordinates(point_0, point_middle, np.abs(rotate_ang))
                    rotate_point_1 = self.Nrotation_angle_get_coor_coordinates(point_1, point_middle, np.abs(rotate_ang))
                else:
                    rotate_point_0 = point_0
                    rotate_point_1 = point_1
                correct_points.append([rotate_point_0, rotate_point_1])
        correct_points = np.array(correct_points)
        final_points = []
        final_points.append(correct_points[0][0])
        for i in range(correct_points.shape[0]-1):
            cur_index = i
            next_index = i + 1 if i < correct_points.shape[0] - 1 else 0
            cur_edge_point_0 = correct_points[cur_index][0]
            cur_edge_point_1 = correct_points[cur_index][1]
            next_edge_point_0 = correct_points[next_index][0]
            next_edge_point_1 = correct_points[next_index][1]
            cur_para_vetr_idx = para_vetr_idxs[cur_index]
            next_para_vetr_idx = para_vetr_idxs[next_index]
            if cur_para_vetr_idx != next_para_vetr_idx:
                L1 = self.line(cur_edge_point_0, cur_edge_point_1)
                L2 = self.line(next_edge_point_0, next_edge_point_1)
                point_intersection = self.intersection(L1, L2)
                final_points.append(point_intersection)
            elif cur_para_vetr_idx == next_para_vetr_idx:
                L1 = self.line(cur_edge_point_0, cur_edge_point_1)
                L2 = self.line(next_edge_point_0, next_edge_point_1)
                marg = self.par_line_dist(L1, L2)
                if marg < 3:
                    point_move = self.point_in_line(next_edge_point_0[0], next_edge_point_0[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                    final_points.append(point_move)
                    correct_points[next_index][0] = point_move
                    correct_points[next_index][1] = self.point_in_line(next_edge_point_1[0], next_edge_point_1[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                else:
                    add_mid_point = (cur_edge_point_1 + next_edge_point_0) / 2
                    add_point_1 = self.point_in_line(add_mid_point[0], add_mid_point[1], cur_edge_point_0[0], cur_edge_point_0[1], cur_edge_point_1[0], cur_edge_point_1[1])
                    add_point_2 = self.point_in_line(add_mid_point[0], add_mid_point[1], next_edge_point_0[0], next_edge_point_0[1], next_edge_point_1[0], next_edge_point_1[1])
                    final_points.append(add_point_1)
                    final_points.append(add_point_2)
        final_points.append(final_points[0])
        final_points = np.array(final_points)
        final_points[:, 1] = h - final_points[:, 1]
        return final_points
    
    def process_images(self):
        folder_path = self.input_folder
        files = os.listdir(folder_path)
        for file in files:
            ori_img1 = cv2.imread(self.input_folder+file)
            ori_img = cv2.medianBlur(ori_img1, 5)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
            ret, ori_img = cv2.threshold(ori_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ori_img, connectivity=8)
            filled_img = np.zeros_like(ori_img)
            for i in range(1, num_labels):
                img = np.zeros_like(labels)
                index = np.where(labels==i)
                img[index] = 255
                img = np.array(img, dtype=np.uint8)
                try:
                    regularization_contour = self.boundary_regularization(img).astype(np.int32)
                except:
                    pass
                cv2.fillPoly(filled_img, [regularization_contour], color=255)

            cv2.imwrite(self.input_folder + file, filled_img)

if __name__ == '__main__':
    # Example usage
    input_folder = 'C:/Users/admin/Downloads/input/'
    regularization = BuildingFootprintRegularization(input_folder)
    regularization.process_images()