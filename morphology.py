# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:04:58 2023

@author: Bassel
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def decorator(fun):
    name=fun.__name__+'.jpg'
    par_dir=os.getcwd()
    path=os.path.join(par_dir,"outputs")
    try:
        os.mkdir(path) 
    except OSError as error:
        pass
    def wrapper(I,size=3, *args, **kwargs):
        rows , cols = I. shape [0:2]
        out= fun(I,size, *args, **kwargs)
        cv2.imshow("org", I)
        cv2.imshow("out", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        outpath=pth(name)
        cv2.imwrite(outpath, out)
        return out
    return wrapper

@decorator
def conv2bn(I , size=0):
    gray= cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    t,bw=cv2.threshold(gray, 225,  255, cv2.THRESH_BINARY_INV)
    return bw


def morph_er(I, size=3, it=1):
    B=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    out=cv2.morphologyEx(I, cv2.MORPH_ERODE, B, iterations=it)
    return out

def pth(name):
    par_dir=os.getcwd()
    return os.path.join(os.path.join(par_dir, "outputs"), name)
 
def morph_di(I,size=3):
    cl=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    return cv2.morphologyEx(I, cv2.MORPH_DILATE, cl)


def morph_close(I,size=3, it=1):
    cl=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    return cv2.morphologyEx(I, cv2.MORPH_CLOSE, cl, iterations=it)

def morph_open(I,size=3, it=1):
    cl=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size,size))
    return cv2.morphologyEx(I, cv2.MORPH_OPEN, cl, iterations=it)

def make_concat(I, fun, size=3):
    rows, cols= I.shape[0], I.shape[1]
    erosions=np.zeros((4*rows, 5*cols) , dtype=np.uint8) # This matrix concatenates all the updates on the photo
    n=20
    j=0
    k=0
    for i in range(n):
        I=fun(I, size)
        if i%5==0 and i!=0:
            j=j+1
            k=0
        erosions[j*rows:rows*(j+1), k*cols: cols*(k+1)]= I
        k=k+1
    return [I, erosions]

def main():
    I=cv2.imread("inputs/balls.jpg")
    rows, cols= I.shape[0], I.shape[1]
    bw=conv2bn(I)
    closed=morph_close(bw,it=5)
    out, erosions = make_concat(closed, morph_er, 7 )
    E=np.zeros_like(out)
    n=6
    closes=np.zeros((n*rows, n*cols) , dtype=np.uint8)
    i=0
    k=0
    j=0
    marg=30
    while not out.all():
        out_ref = morph_close(out)
        if (out_ref==out).all():
            out_ref=morph_di(out, 3)
            outclose= morph_close(out_ref)
            T=outclose-out_ref
        else:
            outclose=out_ref
            T=outclose-out
        
        out = outclose
        E=np.bitwise_or(E, T)
        k=k+1
        if k<marg:
            continue
        if (k-marg)%n==0 and (k-marg)!=0:
            i=0
            j=j+1
        if k <n**2+marg:
            closes[j*rows:rows*(j+1), i*cols:cols*(i+1)]=out
        i=i+1
    res1=E*255+closed
    res=morph_open(res1,size=7,it=22)
    res=morph_er(res, size=3,it=7)
    
    cv2.imshow("Error", E)
    cv2.imshow("res1", res1)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    f_closing = pth("first_closing.jpg")
    f_erosions= pth("erosions.jpg")
    path_closes= pth("closes.jpg")
    path_E= pth("Error.jpg")
    path_c_closed= pth("completely_closed.jpg")
    path_final=pth("final.jpg")
    cv2.imwrite(f_closing, closed)
    cv2.imwrite(f_erosions, erosions)
    cv2.imwrite(path_closes, closes)
    cv2.imwrite(path_E, E)
    cv2.imwrite(path_c_closed, out)
    cv2.imwrite(path_final, res)

if __name__ == "__main__":
    main()