from glob import glob
import cv2, skimage, os
import numpy as np
import random

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def findEssentialMat(self, cur_keyp, prev_keyp, K):
        max_inliers = 0
        inlier_thresh = .01
        H = np.zeros((3,3))

        max_iters = 500
        while max_iters > 0:
            print(max_iters)
            matches = np.random.choice(len(cur_keyp), size = 8, replace = False)

            h = self.secondF(cur_keyp[matches], prev_keyp[matches])
            
            num_inliers = 0
            for i in range(prev_keyp.shape[0]):
                xp = np.array([cur_keyp[i][0], cur_keyp[i][1], 1])
                x = np.array([prev_keyp[i][0], prev_keyp[i][1], 1])
                xhx = abs(xp.T.dot(h).dot(x))
                if xhx <= inlier_thresh:
                    num_inliers+=1

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                H = h

            max_iters -= 1

        return np.matmul(np.matmul(K.T, H), K)

    def secondF(self, cur_keyp, prev_keyp):
        
        cur_mean_x = np.mean(cur_keyp[:, 0])
        cur_mean_y = np.mean(cur_keyp[:, 1])

        prev_mean_x = np.mean(prev_keyp[:, 0])
        prev_mean_y = np.mean(prev_keyp[:, 1])

        cur_center = cur_keyp - np.array([cur_mean_x, cur_mean_y])
        prev_center = prev_keyp - np.array([prev_mean_x, prev_mean_y])

        cur_dist = 0
        prev_dist = 0
        for i in range(cur_center.shape[0]):
            cur_dist += pow(cur_center[i][0]**2 + cur_center[i][1]**2, 1/2)
            prev_dist += pow(prev_center[i][0]**2 + prev_center[i][1]**2, 1/2)
        
        cur_dist /= cur_center.shape[0]
        cur_scale = pow(2,1/2)/cur_dist
        prev_dist /= prev_center.shape[0]
        prev_scale = pow(2,1/2)/ prev_dist

        cur_T = np.array([[cur_scale, 0, -cur_scale*cur_mean_x],
                          [0, cur_scale, -cur_scale*cur_mean_y],
                          [0, 0, 1]])
        prev_T = np.array([[prev_scale, 0, -prev_scale*prev_mean_x],
                          [0, prev_scale, -prev_scale*prev_mean_y],
                          [0, 0, 1]])

        ones = np.ones((cur_keyp.shape[0],1))
        cur_norm = np.hstack([cur_keyp, ones])
        prev_norm = np.hstack([prev_keyp, ones])
        
        cur_norm = np.matmul(cur_T,cur_norm.T).T
        prev_norm = np.matmul(prev_T,prev_norm.T).T

        A = np.zeros((cur_norm.shape[0], 9))
        for i in range(cur_norm.shape[0]):
            xp = cur_norm[i][0]
            yp = cur_norm[i][1]
            x = prev_norm[i][0]
            y = prev_norm[i][1]
            A[i][0] = xp*x
            A[i][1] = xp*y
            A[i][2] = xp
            A[i][3] = yp*x
            A[i][4] = yp*y
            A[i][5] = yp
            A[i][6] = x
            A[i][7] = y
            A[i][8] = 1

        ATA = np.matmul(A.T, A)
        U, S, V = np.linalg.svd(ATA)
        F_init = V[-1:, :].reshape((3,3))

        U, S, V = np.linalg.svd(F_init)
        S[-1] = 0
        S = np.diag(S)
        F = np.matmul(np.matmul(U, S), V)

        F = np.matmul(np.matmul(cur_T.T, F), prev_T)

        return F/F[-1][-1]
    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """

        n = len(self.frames)
        pred_path = np.zeros((n,3))
        
        frame0 = self.imread(self.frames[0])
        
        corner_finder = cv2.FastFeatureDetector_create()
        keypoints = corner_finder.detect(self.imread(self.frames[0]))
        prev_keyp = np.zeros((len(keypoints), 2), dtype=np.float32)
        for i in range(len(keypoints)):
            prev_keyp[i] = keypoints[i].pt

        frame1 = self.imread(self.frames[1])
        keyp2, status, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, prev_keyp, None)
        prev = []
        cur = []
        for i in range(status.shape[0]):
            if status[i][0] == 1:
                prev.append(prev_keyp[i] )
                cur.append(keyp2[i])
        prev_keyp = np.asarray(prev)
        cur_keyp = np.asarray(cur)

        K = np.array([[self.focal_length, 0, self.pp[0]],
                      [0, self.focal_length, self.pp[1]],
                      [0, 0, 1]])

        E, _ = cv2.findEssentialMat(cur_keyp, prev_keyp, focal=self.focal_length, pp=self.pp, method=cv2.RANSAC)
        
        _, cur_R, cur_t, _ = cv2.recoverPose(E, cur_keyp, prev_keyp, focal=self.focal_length, pp = self.pp)

        pred_path[1] = cur_t.T[0]
        prev_keyp = cur_keyp
        
        #print("start")
        for frame_id in range(2, n):
            print(frame_id)
            prev_frame = self.imread(self.frames[frame_id-1])
            cur_frame = self.imread(self.frames[frame_id])
            keyp_cur_id, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, cur_frame, prev_keyp, None)
            prev = []
            cur = []
            for i in range(status.shape[0]):
                if status[i][0] == 1:
                    prev.append(prev_keyp[i])
                    cur.append(keyp_cur_id[i])
            prev_keyp = np.asarray(prev)
            cur_keyp = np.asarray(cur)

            E, _ = cv2.findEssentialMat(cur_keyp, prev_keyp, focal=self.focal_length, pp=self.pp, method=cv2.RANSAC)
            
            _, R, t, _ = cv2.recoverPose(E, cur_keyp, prev_keyp, focal=self.focal_length, pp = self.pp)

            odometry_prediction = cur_R.dot(t)
            cur_t += self.get_scale(frame_id) * odometry_prediction
            cur_R = cur_R.dot(R)
            
            keypoints = corner_finder.detect(self.imread(self.frames[frame_id]))
            cur_keyp = np.zeros((len(keypoints), 2), dtype=np.float32)
            for i in range(len(keypoints)):
                cur_keyp[i] = keypoints[i].pt
            
            pred_path[frame_id] = cur_t.T[0]
            prev_keyp = cur_keyp
        #print("end")

        np.save("predictions.npy",pred_path)
        return pred_path

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)
