# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:32:26 2021

@author: Inés Barbero
"""

import numpy as np
from numpy import arctan2
import math
from math import pi, sin, cos



def MMCC(A,P,K):
    '''Least square fitting'''
    At=np.transpose(A)
    N=np.dot(np.dot(At,P),A)
    T=np.dot(np.dot(At,P),K)
    X=np.dot(np.linalg.pinv(N),T)
    h = len(A)-len(A[0])
    
    Qxx=np.linalg.inv(N)    
    R = np.subtract(np.dot(A, X), K)
    evpu = (np.dot(np.transpose(R),np.dot(P,R)))/h
    Scc = evpu * Qxx
    Var = (np.diag(Scc))
    
    
    return X, Var, R 

def panoramic_resection(point_file, output_file, camera_file, num_iter=10, iterate_weights=False):
    '''Spacial resection of panoramic images, as in Luhmann, 2013
    -point_file contains corresponding points:
        point_number x_img y_img X Y Z
    -camera_file contains pixel size of the image
        x_size, y_size
    Iterate weights can be adjusted so the effects of outliers can be reduced. Useful only with many corresponding points
    The initial camera centre is calculated as centroid of all registration points
    Omega and Phi are initialised to zero
    num_iter:Maximum number of iterations, stops earlier if corrections below 0.1 mm
    
    RETURNS:
    - EOP list: [x0,y0,z0,omega,phi,kappa]
    Also creates outputfile with results'''
        
    sal=open(output_file, 'w')
    #Read coordinates file and store values
    with open(camera_file,'r') as cam:
        cam=cam.readlines()
        x_size=cam[0].split()[0]
        y_size=cam[0].split()[1]
    x_size=17054
    y_size=8527
    x_img=[]
    y_img=[]
    x_ter=[]
    y_ter=[]
    z_ter=[]
    kap_img=[]
    with open(point_file,'r') as coord:
        for line in coord:
            line=line.split(' ')
            x_img.append(float(line[1]))
            y_img.append(y_size - float(line[2]))
            x_ter.append(float(line[3]))
            y_ter.append(float(line[4]))
            z_ter.append(float(line[5]))
            kap_img.append(float(line[1])*2*math.pi/x_size)
                
    #Approximate parameters
    r = x_size/(2*math.pi)
    
    '''Initial parameters'''
    #Approximate camera coordinates as centre between points
    xl = np.mean(x_ter)
    yl = np.mean(y_ter)
    zl = np.mean(z_ter)
    
    phi = 0
    ome = 0
    kap_pt_1_img = kap_img[0]         
    az_pt_1 = np.arctan2((y_ter[0]-yl),(x_ter[0]-xl))
    kap = az_pt_1 - kap_pt_1_img - math.pi/2  #Azimut
    kap=166*math.pi/180
        
    
    #Collinearity equations for cylindrical images, as in Luhmann
    # cylindrical_x= -r_val*(arctan2((m12*(x-xl)+m22*(y-yl)+m32*(z-zl)),(m11*(x-xl)+m21*(y-yl)+m31*(z-zl))))
    # cylindrical_y= (0.5*y_size) + (r_val/(((x-xl)**2+(y-yl)**2)**0.5))*(m13*(x-xl)+m23*(y-yl)+m33*(z-zl))
        
    for i in range (num_iter):
        A = []
        K = []
        
        #Rotation matrix...
        m11=cos(phi)*cos(kap)
        m12=-cos(phi)*sin(kap)
        m13=sin(phi)
        
        m21=cos(ome)*sin(kap)+sin(ome)*sin(phi)*cos(kap)
        m22=cos(ome)*cos(kap)-sin(ome)*sin(phi)*sin(kap)
        m23=-sin(ome)*cos(phi)
        
        m31=sin(ome)*sin(kap)-cos(ome)*sin(phi)*cos(kap)
        m32=sin(ome)*cos(kap)+cos(ome)*sin(phi)*sin(kap)
        m33=cos(ome)*cos(phi)
    
        j=0
        #For each point         
        while j<len(x_ter):
            x=x_ter[j]
            y=y_ter[j]
            z=z_ter[j]
            xCal = -r*(np.arctan2(m12*(x-xl)+m22*(y-yl)+m32*(z-zl),m11*(x-xl)+m21*(y-yl)+m31*(z-zl)))
            # xCal2 = -int((np.arctan2((y-yl),(x-xl)))*r)
            
            yCal =  (0.5*y_size) + (r/(((x-xl)**2+(y-yl)**2)**0.5))*(m13*(x-xl)+m23*(y-yl)+m33*(z-zl))
            # yCal2 = ((0.5*y_size)+(((z-zl) * (r/(((x-xl)**2+(y-yl)**2)**0.5)))))
            
            if xCal<0:
                xCal=xCal+x_size
            # if xCal2<0:
            #     xCal2=xCal2+x_size
            
            #Calculate B parameters, which are the derivates of Fx and Fy in function of each parameter, the compose matrix A of LS fitting
    
            B11 = -r*(-((x - xl)*sin(kap)*cos(phi) - (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) - (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))*cos(kap)*cos(phi)/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2) + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))*sin(kap)*cos(phi)/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2)) 
            B12 = -r*((-sin(kap)*cos(ome) - sin(ome)*sin(phi)*cos(kap))*((x - xl)*sin(kap)*cos(phi) - (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) - (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2) + (sin(kap)*sin(ome)*sin(phi) - cos(kap)*cos(ome))*((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2))
            B13 = -r*((-sin(kap)*sin(ome) + sin(phi)*cos(kap)*cos(ome))*((x - xl)*sin(kap)*cos(phi) - (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) - (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2) + (-sin(kap)*sin(phi)*cos(ome) - sin(ome)*cos(kap))*((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2))  
            
            B14 = -r*(((y - yl)*(-sin(kap)*sin(ome) + sin(phi)*cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)))*((x - xl)*sin(kap)*cos(phi) - (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) - (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2) + ((y - yl)*(-sin(kap)*sin(phi)*cos(ome) - sin(ome)*cos(kap)) + (z - zl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)))*((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2))
            B15 = -r*((-(-x + xl)*sin(kap)*sin(phi) - (y - yl)*sin(kap)*sin(ome)*cos(phi) + (z - zl)*sin(kap)*cos(ome)*cos(phi))*((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2) + ((x - xl)*sin(kap)*cos(phi) - (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) - (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))*(-(x - xl)*sin(phi)*cos(kap) + (y - yl)*sin(ome)*cos(kap)*cos(phi) - (z - zl)*cos(kap)*cos(ome)*cos(phi))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2))
            B16 = -r*(((-x + xl)*cos(kap)*cos(phi) + (y - yl)*(-sin(kap)*cos(ome) - sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(-sin(kap)*sin(ome) + sin(phi)*cos(kap)*cos(ome)))*((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2) + (-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))*((x - xl)*sin(kap)*cos(phi) - (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) - (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))/((-(x - xl)*sin(kap)*cos(phi) + (y - yl)*(-sin(kap)*sin(ome)*sin(phi) + cos(kap)*cos(ome)) + (z - zl)*(sin(kap)*sin(phi)*cos(ome) + sin(ome)*cos(kap)))**2 + ((x - xl)*cos(kap)*cos(phi) + (y - yl)*(sin(kap)*cos(ome) + sin(ome)*sin(phi)*cos(kap)) + (z - zl)*(sin(kap)*sin(ome) - sin(phi)*cos(kap)*cos(ome)))**2))
            
            B21 = r*(1.0*x - 1.0*xl)*((x - xl)**2 + (y - yl)**2)**(-1.5)*((x - xl)*sin(phi) - (y - yl)*sin(ome)*cos(phi) + (z - zl)*cos(ome)*cos(phi)) - r*((x - xl)**2 + (y - yl)**2)**(-0.5)*sin(phi)
            B22 = r*(1.0*y - 1.0*yl)*((x - xl)**2 + (y - yl)**2)**(-1.5)*((x - xl)*sin(phi) - (y - yl)*sin(ome)*cos(phi) + (z - zl)*cos(ome)*cos(phi)) + r*((x - xl)**2 + (y - yl)**2)**(-0.5)*sin(ome)*cos(phi)
            B23 = -r*((x - xl)**2 + (y - yl)**2)**(-0.5)*cos(ome)*cos(phi)
            
            B24 = r*((-y + yl)*cos(ome)*cos(phi) - (z - zl)*sin(ome)*cos(phi))*((x - xl)**2 + (y - yl)**2)**(-0.5)
            B25 = r*((x - xl)**2 + (y - yl)**2)**(-0.5)*((x - xl)*cos(phi) - (-y + yl)*sin(ome)*sin(phi) - (z - zl)*sin(phi)*cos(ome))
            B26 = 0
            
            Fx = float(x_img[j] - xCal)
            Fy = float(y_img[j] - yCal)
            


            
            # #Se añaden dos ecuaciones MMCC por punto
            A.append([float(B14), float(B15), float(B16), float(B11), float(B12), float(B13)])
            A.append([float(B24), float(B25), float(B26), float(B21), float(B22), float(B23)])
            K.append([float(-Fx)])
            K.append([float(-Fy)])
            j+=1
            
        #Least squares
        p = [1] * len(A)#Begin with all weights=1
        P=np.diag(p) 
       
        if iterate_weights:
            if i==0:
                p = [1] * len(A)#Begin with all weights=1
                P=np.diag(p) 
            else:
                p=1/abs(np.array(R))
                p_flat = [float(item) for sublist in p for item in sublist]
                P=np.diag(p_flat)
            
        else:
            p = [1] * len(A)#Begin with all weights=1
            P=np.diag(p) 

        X, Var, R = MMCC(A,P,K)
        
    
        ome = (ome - X[0,0])
        phi = (phi - X[1,0])
        kap = (kap - X[2,0])
        xl = (xl - X[3,0])
        yl = (yl - X[4,0])
        zl = (zl - X[5,0])
        
        #Print results to output file:
        sal.write("---------------  Iteration: "+str(i)+'  -------------------\n\n')
        sal.write("Corrections: "+'\n\n')
        sal.write("dXL = " + str(X[3,0]) + '\n'+ "dYL = " + str(X[4,0]) + '\n' + "dZL="+ str(X[5,0])+'\n\n' ) 
        sal.write( "dOmega = "+str((X[0,0])*180/math.pi)+'\n'+ "dPhi = " + str((X[1,0])*180/math.pi) + '\n' + "dKappa="+ str((X[2,0])*180/math.pi) +'\n\n')
        sal.write("Obtained parameters: "+'\n\n')
        sal.write("XL = " + str(xl) + '\n'+ "YL = " + str(yl) + '\n' + "ZL="+ str(zl)+'\n\n' ) 
        sal.write( "Omega = "+str((ome)*180/math.pi)+'\n'+ "Phi = " + str((phi)*180/math.pi) +'\n' + "Kappa="+ str((kap)*180/math.pi) +'\n\n')
        if abs(X[0,0])<0.0001 and abs(X[1,0])<0.0001 and abs(X[2,0])<0.0001 and abs(X[3,0])<0.0001 and abs(X[4,0])<0.0001 and abs(X[5,0])<0.0001:
            break
            
    sal.close()
    
    return [xl,yl,zl,ome,phi,kap]

output_file = 'Results_ori.sal'
point_file = 'coord_ori.txt'
camera_file = 'camera.txt'
EOP = panoramic_resection(point_file, output_file, camera_file, num_iter=10, iterate_weights=False)
print(EOP)