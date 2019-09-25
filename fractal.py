# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 19:25:12 2019

@author: uhlse
"""

import cv2 as cv2
import numpy as np
import time

path = "/"
#highres: to generate high-res picture at pos efined by moveX, moveY, zoom
#lowres: to quickly preview and explore the fractal in a window 
#       use q to quit.       
#       use mouse clicks to zoom in and out. 
#       W, A, S, D to move presicly. 0 to snap HD pic
#       see code for more
highRes_not_lowRes = False      # interactiv low res
#highRes_not_lowRes = True       # high res
single_pic_not_autoZoom = True    # single pic 
#single_pic_not_autoZoom = False    # autozoom 


#define the starting position
#moveX, moveY, zoom = 0.04958839852650198, -0.7494709910455487, 22984174.518310547      #big spiral(high iteration)
#moveX, moveY, zoom =  0.9623177404464485 ,  -0.08850052965329175 ,  4905769.773057054  # little mandelbrot background
#moveX, moveY, zoom =  0.9623179755811408 ,  -0.08850059288835598 ,  19623079.092228215 #big rip
#moveX, moveY, zoom =  0.9623179770918335 ,  -0.08850059424492188 ,  13402827.055684863 #lil mandelbrot 2
#moveX, moveY, zoom =  0.9623180203267551 ,  -0.08850058804711496 ,  270976764.3514345 #shadow monster
#moveX, moveY, zoom =  -0.9230308085322426 ,  -0.10620280408094843 ,  5407304232  #double spider eye
#moveX, moveY, zoom = 0.0, -0.45, 1
#moveX, moveY, zoom =  -0.4580452099442482 ,  -0.6702441970507304 ,  67108864
#moveX, moveY, zoom =  1.0396951371156147 ,  -0.15284673542690733 ,  268435456.0
#moveX, moveY, zoom =  1.0396951378063455 ,  -0.15284673572764693 ,  17179869184.0
moveX, moveY, zoom =  0 ,  0 ,  1

# define amount of iterations = 255 * depth - 1
depth = 5 

#to choose between Mandelbrot fractal an Julia set. Mandelbrot is much faster
mandelbrot_not_juliaset = True

#define the julia set
cX, cY = -0.7, 0.27015 #original
#cX, cY = -0.7, 0.47015 #stars
#cX, cY = -0.7, 0.27115
#cX, cY = 0, 0.7
#cX, cY = -0.71, -0.27215

w, h = 284,160
ix,iy = -1,-1
pic_count = 0
    

def main():
    #test_gradient()
    global highRes_not_lowRes, single_pic_not_autoZoom, ind
    #highResSetWalk()
    #return###################################################################
    print()
    ind=getIndex() # unique identifier
    if highRes_not_lowRes:
        if single_pic_not_autoZoom:
            print("snap one high res pic\n")
            highResPic()
        else:
            print("initiate high res autozoom\n")
            highResAutoZoom()
    else:
        print("low res interactiv\n")
        lowResInteractive()
    
    return

def frac(res = 0, direction = 1.0, index=0):
    global ix,iy, moveX, moveY, w, h, zoom, pic_count, cX, cY, depth
    
    if res == 0: # HD quality
        w, h = 1920,1080
    elif res == 1: # preview
        w, h = 284,160 
    elif res == 2: # preview, no save!
        w, h = 284,160 
    elif res == 4: # 4k
        w, h = 3840, 2160 
    elif res == 5: # stupid high res
        w, h = 8889, 5000
    
        
    
    exit_times = np.zeros((h,w), np.uint8)-1 #black
    
    pic_count += 1
    
    
    if mandelbrot_not_juliaset:  #mandelbrot
        dataname = path+"mndlbrt"+"_i{:01}".format(index)+"_c{:04}".format(pic_count)+"_r{:01}".format(res)+"_z{:016}".format(zoom)+"_x"+str(moveX)+"_y"+str(moveY)+".png"
        print("\n" + "count: {:04}".format(pic_count)+" r: {:01}".format(res)+" zoom: {:}".format(zoom))
        maxIter = 256*depth - 1
        x0 = (w/h)*(0 - w/2)/(0.5*zoom*w) + moveX
        x1 = (w/h)*(w - w/2)/(0.5*zoom*w) + moveX
        y0 = (0 - h/2)/(0.5*zoom*h) + moveY 
        y1 = (h - h/2)/(0.5*zoom*h) + moveY 
        X = np.linspace(x0, x1, w)
        Y = np.linspace(y0, y1, h)
         
        #broadcast X to a square array
        C = Y[:, None] + 1J * X
        #initial value is always zero
        Z = np.zeros_like(C)
         
        exit_times = maxIter * np.ones(C.shape, np.uint16)
        mask = exit_times > 0
         
        
        for k in range(maxIter):
            Z[mask] = Z[mask] * Z[mask] + C[mask]
            mask, old_mask = abs(Z) < 2, mask
            #use XOR to detect the area which has changed 
            exit_times[mask ^ old_mask] = k
            
            #delete_me = np.zeros(C.shape, np.uint8)-1
            #delete_me = delete_me & (mask)
            #cv2.imshow('test', delete_me*255)
            #cv2.waitKey(1)
            
            #print(" ", k, end="")
            if k % (maxIter//40) == 0:
                print("|", end="")   
        #exit_times[mask] = abs(Z)[mask] * 6000 ## Inverse/inner Mandelbrot set
        print("")
            
    if False:  #julia set
        maxIter = 256*1 - 1
        dataname = path+"jli-st"+"_i{:01}".format(index)+"_c{:04}".format(pic_count)+"_r{:01}".format(res)+"_z{:016}".format(zoom)+"_cx"+str(cX)+"_cy"+str(cY)+"_x"+str(moveX)+"_y"+str(moveY)
        print("\n" + dataname)
        for x in range(w):
            for y in range(h):
                zx = (w/h)*(x - w/2)/(0.5*zoom*w) + moveX
                zy = 1.0*(y - h/2)/(0.5*zoom*h) + moveY 
                i = maxIter 
                while zx*zx + zy*zy < 4 and i > 1: 
                    tmp = zx*zx - zy*zy + cX 
                    zy,zx = 2.0*zx*zy + cY, tmp 
                    i -= 1
      
                # convert byte to RGB (3 bytes), kinda  
                # magic to get nice colors 
                exit_times[y,x] = i
            if x%(w/10) == 0:
                print(100*x/w, "%")

    
    #exit_times = cv2.applyColorMap(exit_times, cv2.COLORMAP_JET)  #COLORMAP_JET  COLORMAP_HOT 
    
    if not res == 2 and not res == 1:
        for i in range(1,4): #1,2,3
            temp = (exit_times/i)%256
            temp = cv2.convertScaleAbs(temp)
            out = applyColorMap(temp, 'infrared')
            cv2.imwrite(dataname+"_m1"+str(i)+"+.png", out)   
            out = applyColorMap(-temp, 'infrared')
            cv2.imwrite(dataname+"_m1"+str(i)+"-.png", out)   
            out = applyColorMap(temp, 'pastell')
            cv2.imwrite(dataname+"_m2"+str(i)+"+.png", out)   
            out = applyColorMap(-temp, 'pastell')
            cv2.imwrite(dataname+"_m2"+str(i)+"-.png", out)   
            out = cv2.applyColorMap(temp, cv2.COLORMAP_JET)  #COLORMAP_JET  COLORMAP_HOT 
            cv2.imwrite(dataname+"_m3"+str(i)+"+.png", out)   
            out = cv2.applyColorMap(-temp, cv2.COLORMAP_JET)  #COLORMAP_JET  COLORMAP_HOT 
            cv2.imwrite(dataname+"_m3"+str(i)+"-.png", out)  
            out = cv2.applyColorMap(temp, cv2.COLORMAP_HOT)  #COLORMAP_JET  COLORMAP_HOT 
            cv2.imwrite(dataname+"_m4"+str(i)+"+.png", out)   
            out = cv2.applyColorMap(-temp, cv2.COLORMAP_HOT)  #COLORMAP_JET  COLORMAP_HOT 
            cv2.imwrite(dataname+"_m4"+str(i)+"-.png", out)   
            out = applyColorMap(temp, 'crab')
            cv2.imwrite(dataname+"_m5"+str(i)+"+.png", out)   
            out = applyColorMap(-temp, 'crab')
            cv2.imwrite(dataname+"_m5"+str(i)+"-.png", out)
    
    exit_times = (exit_times/1)%256
    exit_times = cv2.convertScaleAbs(exit_times)
    #exit_times = cv2.applyColorMap(exit_times, cv2.COLORMAP_JET)  #COLORMAP_JET  COLORMAP_HOT 
    exit_times = applyColorMap(-exit_times, 'infrared')
    if res == 1:
        cv2.imwrite(dataname+".png", exit_times)  
    return exit_times

def getIndex():
    return int(time.time())%100000

def highResPic():
    global moveX, moveY, zoom, ind
    
    #different julia set
    #cX, cY = -0.71, -0.27215
    #index_f, moveX, moveY, zoom = getIndex(), 0, 0, 1
    #index_f, moveX, moveY, zoom = 7 , -0.5031048082723688,-0.1666950187800161, 1


    if not get_low_res_preview(): #at fully zoomed state
        print("rejected")
        return

    return frac(res=5, index=ind)

def highResAutoZoom():
    global moveX, moveY, zoom

 
    #index_f,z_init,z_step,moveX,moveY = getIndex(),0.3333,  3,-0.09652482073760113,0.02781022517143311
    #index_f,z_init,z_step,moveX,moveY = getIndex(),  0.25,  2, -1.4829346051729473,0.13718436154146357
    #index_f,z_init,z_step,moveX,moveY = getIndex(),  0.4 ,  2, 0.07613367127375358,-0.050875771980016096
    #index_f,z_init,z_step,moveX,moveY = getIndex(),   0.3 ,1.1, 0.19108351225330675,0.09889883393665763
    #frac_i0_c0035_r1_z617673396283947.0_x-0.5031048082723688_y-0.1666950187800161
    
    #different julia set
    #cX, cY = -0.71, -0.27215
    #index_f,z_init,z_step,moveX,moveY = getIndex(),   0.1 ,1.1, -0.5031048082723688,-0.1666950187800161
    
    #sterne -0.7, 0.47015 
    #index_f,z_init,z_step,moveX,moveY = getIndex(),   0.1 ,1.1, 0.5165192006363556,-0.23947614344799617
    
    #cX, cY = -0.71, -0.27215
    #index_f,z_init,z_step,moveX,moveY = getIndex(),   0.3 ,1.2, 0.38173387697908273,-0.001957012868220015
    
    #mandelbrot
    #mndlbrt_i85358_c0116_r1_z617673396283947.0_x-0.0695546984114432_y-0.7489455384383931
    #index_f,z_init,z_step,moveX,moveY = getIndex(),   0.3 ,1.2, -0.0695546984114432,-0.7489455384383931
    
    #iter >5*
    index_f,z_init,z_step,moveX,moveY = getIndex(),   0.3 ,1.2, -0.923030808469231,-0.10620280412518017
    
    
    zoom = z_init
    if not get_low_res_preview(): #at fully zoomed state
        print("rejected")
        return
    
    zoom = 100000000000000
    if not get_low_res_preview(): #at fully zoomed state
        print("rejected")
        return
    
    zoom = z_init
    zoom_faktor = z_step 
    
    while zoom < 100000000000000:
        frac(res=0, index=index_f)
        zoom *= zoom_faktor

#for juliaset, to coninously change the julia set.
def highResSetWalk():
    global moveX, moveY, zoom, cX, cY
    
    cX, cY = -0.7, 0.27015 #original
    
    index_f,moveX,moveY,zoom = getIndex(),0,0,1
    cY_start, cY_end, num = 0.25015, 0.33415, 1000
    
    cY=cY_start
    if not get_low_res_preview(): #at fully zoomed state
        print("rejected")
        return
    cY=cY_end
    if not get_low_res_preview(): #at fully zoomed state
        print("rejected")
        return
    
    step_cY = (cY_end-cY_start)/num
    cY=cY_start
    for i in range(num):
        frac(res=0, index=index_f)
        cY += step_cY

def get_low_res_preview():
    cv2.namedWindow('Preview')
    pix = frac(2)
    cv2.imshow('Preview', pix)
    k = cv2.waitKey(0) & 0xFF
    ret = True
    if k == 27:
        ret = False
    elif k == ord('q'):
        ret = False
    cv2.destroyAllWindows()
    return ret

def get_new_pos(ix,iy, moveX, moveY, w, h, zoom, direction, movement=0):
    if movement == 0:
        if ix == -1 and iy == -1:
            return 0, 0, 1
        new_moveX = (w/h)*(ix - w/2)/(0.5*zoom*w) + moveX
        new_moveY = 1.0*(iy - h/2)/(0.5*zoom*h) + moveY 
        
        zoom = zoom*direction
        return new_moveX, new_moveY, zoom
    elif movement == 1: #north
        new_moveY = 1.0*(-h/10)/(0.5*zoom*h) + moveY 
        new_moveX = moveX
        return new_moveX, new_moveY, zoom
    elif movement == 3: #south
        new_moveY = 1.0*(h/10)/(0.5*zoom*h) + moveY 
        new_moveX = moveX
        return new_moveX, new_moveY, zoom
    elif movement == 2: #east
        new_moveX = (w/h)*(w/10)/(0.5*zoom*w) + moveX
        new_moveY = moveY
        return new_moveX, new_moveY, zoom
    elif movement == 4: #west
        new_moveX = (w/h)*(-w/10)/(0.5*zoom*w) + moveX
        new_moveY = moveY
        return new_moveX, new_moveY, zoom

resize_faktor = 1.5 #for GUI Window

def mouse_event(event,x,y,flags,param):
    global ix,iy,pix,moveX, moveY, zoom, w, h, ind, resize_faktor
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x/resize_faktor,y/resize_faktor
        moveX, moveY, zoom = get_new_pos(ix,iy, moveX, moveY, w, h, zoom, 4)
        pix = frac(res=1, index=ind)
    if event == cv2.EVENT_RBUTTONDOWN:
        ix,iy = x,y
        zoom /= 4
        pix = frac(res=1, index=ind)
    if event == cv2.EVENT_MBUTTONDOWN:
        ix,iy = x,y
        zoom = 1
        pix = frac(res=1, index=ind)




def lowResInteractive():
    cv2.namedWindow('Fractal')
    cv2.setMouseCallback('Fractal',mouse_event)
    global pix, ind, resize_faktor
    global moveX, moveY, zoom, w, h
    pix = frac(1, index=ind)
    
    while(1):
        if w < 500:
            cv2.imshow('Fractal', cv2.resize(pix, (int(w*resize_faktor),int(h*resize_faktor))))
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('q'):
            break
        elif k == ord('p'):
            print("moveX, moveY, zoom = ",moveX,", ",moveY,", ",zoom)
        elif k == ord('0'):
            print("\nprocessing HD picture, takes a few min")
            pix = frac(res=0, index=ind)
            print("saved HD picture")
        elif k == ord('4'):
            print("\nprocessing 4k picture, takes a few min")
            pix = frac(res=4, index=ind)
            print("saved 4k picture")
        elif k == ord('5'):
            print("\nprocessing 8k picture, takes a few min")
            pix = frac(res=5, index=ind)
            print("saved 8k picture")
        elif k == ord('w'):
            moveX, moveY, zoom = get_new_pos(ix,iy, moveX, moveY, w, h, zoom, 1, 1)
            pix = frac(res=1, index=ind)
        elif k == ord('d'):
            moveX, moveY, zoom = get_new_pos(ix,iy, moveX, moveY, w, h, zoom, 1, 2)
            pix = frac(res=1, index=ind)
        elif k == ord('s'):
            moveX, moveY, zoom = get_new_pos(ix,iy, moveX, moveY, w, h, zoom, 1, 3)
            pix = frac(res=1, index=ind)
        elif k == ord('a'):
            moveX, moveY, zoom = get_new_pos(ix,iy, moveX, moveY, w, h, zoom, 1, 4)
            pix = frac(res=1, index=ind)
        elif k == ord('t'):
            zoom *= 1.1
            pix = frac(res=1, index=ind)
        elif k == ord('g'):
            zoom /= 1.1
            pix = frac(res=1, index=ind)
    cv2.destroyAllWindows()
    

#def test_gradient():
#    gradient = np.zeros((255,200), np.uint8)-1 #black
#    for i in range(255):
#        for j in range(200):
#            gradient[i,j] = i
#    gradient = applyColorMap(gradient, 'infrared')
#    for i in range(255):
#        for j in range(100,200):
#            gradient[i,j] = i
#    cv2.imshow('grad', gradient)
#    cv2.waitKey(1000)

def rgb_tuple(h):
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def applyColorMap(gray, cmap):
    mx = 256  # if gray.dtype==np.uint8 else 65535
    lut = np.empty(shape=(256, 3))
    if(cmap == 'flame'):
        cmap = (
            # taken from pyqtgraph GradientEditorItem
            (0  , (0, 0, 0)),
            (0.2, (7, 0, 220)),
            (0.5, (236, 0, 134)),
            (0.8, (246, 246, 0)),
            (1.0, (255, 255, 255))
        )
    elif(cmap == 'pastell'):
        cmap = (
            # taken from pyqtgraph GradientEditorItem
            (0,   rgb_tuple("FFCCF9")),
            (0.2, rgb_tuple("D5AAFF")),
            (0.4, rgb_tuple("B5B9FF")),
            (0.6, rgb_tuple("85E3FF")),
            (0.8, rgb_tuple("F3FFE3")),
            (1.0, rgb_tuple("FFBEBC"))
        )
    elif(cmap == 'houses'):
        cmap = (
            # taken from pyqtgraph GradientEditorItem
            (0,   rgb_tuple("70ae98")),
            (0.2, rgb_tuple("ecbe7a")),
            (0.6, rgb_tuple("e58b88")),
            (0.8, rgb_tuple("9dabdd")),
            (1.0, rgb_tuple("d9effc"))
        )
    elif(cmap == 'beach'):
        cmap = (
            # taken from pyqtgraph GradientEditorItem
            (0,   rgb_tuple("dfc7c1")),
            (0.2, rgb_tuple("f4dcd6")),
            (0.6, rgb_tuple("b2d9ea")),
            (0.8, rgb_tuple("84b4c8")),
            (1.0, rgb_tuple("619196"))
        )
    elif(cmap == 'crab'):
        cmap = (
            # taken from pyqtgraph GradientEditorItem
            (0,   rgb_tuple("ab6393")),
            (0.2, rgb_tuple("9c8ade")),
            (0.6, rgb_tuple("e69288")),
            (0.8, rgb_tuple("feb68e")),
            (1.0, rgb_tuple("e8ccc0"))
        )
    elif(cmap == 'infrared'):
        cmap = (
            (0,    rgb_tuple("ffffff")),
            (0.14, rgb_tuple("ffcb03")),
            (0.26, rgb_tuple("f77e00")),
            (0.43, rgb_tuple("e0363e")),
            (0.57, rgb_tuple("bd0398")),
            (0.71, rgb_tuple("74019e")),
            (0.86, rgb_tuple("190081")),
            (1,    rgb_tuple("000212"))
        )
    # build lookup table:
    lastval, lastcol = cmap[0]
    for step, col in cmap[1:]:
        val = int(step * mx)
        for i in range(3):
            lut[lastval:val, i] = np.linspace(
                lastcol[i], col[i], val - lastval)

        lastcol = col
        lastval = val

    s0, s1 = gray.shape
    out = np.empty(shape=(s0, s1, 3), dtype=np.uint8)

    for i in range(3):
        out[..., i] = cv2.LUT(gray, lut[:, 2-i])
    return out 

main()
