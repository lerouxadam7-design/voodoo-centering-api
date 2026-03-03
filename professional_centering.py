import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    def line_intersection(self, line1, line2):
        x1,y1,x2,y2 = line1
        x3,y3,x4,y4 = line2

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if denom == 0:
            return None

        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
        return [px, py]

    def perspective_warp(self, image, pts):

        rect = np.array(pts, dtype="float32")

        s = rect.sum(axis=1)
        tl = rect[np.argmin(s)]
        br = rect[np.argmax(s)]

        diff = np.diff(rect, axis=1)
        tr = rect[np.argmin(diff)]
        bl = rect[np.argmax(diff)]

        rect = np.array([tl, tr, br, bl], dtype="float32")

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0,0],
            [maxWidth-1,0],
            [maxWidth-1,maxHeight-1],
            [0,maxHeight-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    def detect_slab_rectangle(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 200)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi/180,
            threshold=150,
            minLineLength=int(image.shape[1]*0.5),
            maxLineGap=20
        )

        if lines is None:
            return None

        lines = lines[:,0,:]

        vertical = []
        horizontal = []

        for x1,y1,x2,y2 in lines:
            if abs(x1-x2) < 20:
                vertical.append([x1,y1,x2,y2])
            if abs(y1-y2) < 20:
                horizontal.append([x1,y1,x2,y2])

        if len(vertical) < 2 or len(horizontal) < 2:
            return None

        left_line = min(vertical, key=lambda l: min(l[0],l[2]))
        right_line = max(vertical, key=lambda l: max(l[0],l[2]))
        top_line = min(horizontal, key=lambda l: min(l[1],l[3]))
        bottom_line = max(horizontal, key=lambda l: max(l[1],l[3]))

        tl = self.line_intersection(left_line, top_line)
        tr = self.line_intersection(right_line, top_line)
        br = self.line_intersection(right_line, bottom_line)
        bl = self.line_intersection(left_line, bottom_line)

        if None in [tl,tr,br,bl]:
            return None

        return [tl,tr,br,bl]

    def calculate_centering(self, cropped):

        h,w = cropped.shape[:2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150)

        coords = np.column_stack(np.where(edges>0))

        if coords.size == 0:
            return 0.5,0.5,0.0

        y_min,x_min = coords.min(axis=0)
        y_max,x_max = coords.max(axis=0)

        left_margin = x_min
        right_margin = w - x_max
        top_margin = y_min
        bottom_margin = h - y_max

        if max(left_margin,right_margin)==0 or max(top_margin,bottom_margin)==0:
            return 0.5,0.5,0.0

        h_ratio = min(left_margin,right_margin)/max(left_margin,right_margin)
        v_ratio = min(top_margin,bottom_margin)/max(top_margin,bottom_margin)

        return float(h_ratio),float(v_ratio),1.0

    def analyze_array(self,image_array):

        target_width = 1200
        h,w = image_array.shape[:2]
        scale = target_width/w
        image = cv2.resize(image_array,(target_width,int(h*scale)))

        slab_pts = self.detect_slab_rectangle(image)

        if slab_pts is None:
            return {"horizontal_ratio":0.5,"vertical_ratio":0.5,"confidence":0.0}

        slab_warped = self.perspective_warp(image, slab_pts)

        # crop inward to remove plastic border
        crop_pct = 0.07
        h2,w2 = slab_warped.shape[:2]

        x_start = int(w2*crop_pct)
        x_end = int(w2*(1-crop_pct))
        y_start = int(h2*crop_pct)
        y_end = int(h2*(1-crop_pct))

        cropped = slab_warped[y_start:y_end, x_start:x_end]

        h_ratio,v_ratio,confidence = self.calculate_centering(cropped)

        return {
            "horizontal_ratio":h_ratio,
            "vertical_ratio":v_ratio,
            "confidence":confidence
        }
