import _pickle as cPickle
import numpy as np
import pandas as pd
import cv2
import time

def round(h,w):
    h -= h % -50
    w -= w % -50
    return h,w
def compare(f,s,distance):
     if abs(f[0]-s[0]) <= distance:
         if abs(f[1]-s[1]) <= distance:
             return 1
     elif abs(f[1]-s[1]) <= distance:
         if abs(f[0]-s[0]) <= distance:
             return 1
     else:
         return 0
def main():
    thresh = 150 #set thresh for black and white sensitivity(higher for less sensitivity)
    spc = 5 #spacing between each cropping
    dist = 40 #detect overlap for same numbers
    overlap = 10 # detect overlap between diff numbers
    ratio = [(30,30),(40,40)] #ratio for cropping

    img = cv2.imread("nums.png")
    gray = ~cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, bwimg) = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    h,w,_ =img.shape
    h,w = round(h,w)
    rs = cv2.resize(bwimg,(w,h),interpolation = cv2.INTER_AREA)
    pool = []
    cord = []
    for r1,r2 in ratio:
        for i in range(int(h/spc)-1):
            for j in range(int(w/spc)-1):
                y=i*spc; x=j*spc;
                if y==h:
                    y=0
                if x==w:
                    x=0
                crp = rs[y:y+r1, x:x+r2]
                crp = cv2.resize(crp,(28,28),interpolation = cv2.INTER_AREA)
                mask = np.ones(crp.shape[:2], dtype = "uint8")
                cv2.rectangle(mask, (2,2),(crp.shape[1]-2,crp.shape[0]-2), 0, -1)
                bd = cv2.bitwise_and(crp, crp, mask = mask)
                if cv2.countNonZero(crp)==0:
                    continue
                else:
                    if cv2.countNonZero(bd)==0:
                        array = np.array(crp)
                        ravel = array.ravel()
                        lst = ravel.tolist()
                        lst = ravel.reshape(1,-1)
                        pool.append(lst)
                        cord.append([x,y])
                    else:
                        continue

    with open('pickle.sav', 'rb') as f:
        clf = cPickle.load(f)
    info = []
    for index,content in enumerate(pool):
        prob=clf.predict_proba(content)
        score=max(prob.ravel())
        predict = clf.predict(content)
        if score >= 0.5:
            info.append((predict,score,cord[index]))
    fpool=[]
    gpool=[]
    for k,target in enumerate(info):
        check = []
        if target in fpool:
            continue
        for l, task in enumerate(info):
            if l == k:
                continue
            if target[0] == task[0]:
                if compare(target[2],task[2],dist) == 1:
                    check.append(task)
                    fpool.append(task)
        if not check:
             check.append(target)
        fscore = [m[1] for m in check]
        index = fscore.index(max(fscore))
        gpool.append(check[index])
    for n in gpool:
        for o in gpool:
            if n == o:
                continue
            if compare(n[2],o[2],overlap) == 1:
                if n[1]>= o[1]:
                    gpool.remove(o)
                else:
                    gpool.remove(n)
    for p in gpool:
        fx = p[2][0]
        fy = p[2][1]
        text = p[0]
        cv2.rectangle(img,(fx-10,fy-10),(fx+30,fy+30),(fx,fy))
        cv2.putText(img,str(text), (fx,fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)#cv2.imshow("output",img)

    cv2.imshow("output",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
