import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.misc import imread
from skimage.feature import canny
from scipy.ndimage.filters import sobel
from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

def gradient_orientation(image):
    '''
    Calculate the gradient orientation for edge point in the image
    '''
    #scipy.ndimage.sobel
    dx = sobel(image, axis=0, mode='constant')
    dy = sobel(image, axis=1, mode='constant')
    dz = sobel(image, axis=1, mode='constant')
    
    #For 3D instead of a single gradient value, we need two angles that define a normal vector
    #Phi is the angle between the positive x-axis to the projection of the normal vector the x-y plane (around +z)
    #Psi is the angle between the positive z-axis to the normal vector
    
    phi = np.arctan2(dy ,dx) * 180 / np.pi
    psi = np.arctan2(np.sqrt(dx*dx + dy*dy), dz) * 180 / np.pi
    
    
    print("dx: ", dx.shape)
    print("dy: ", dy.shape)
    print("phi: ", phi.shape)
    print("psi:", psi.shape)
    print(np.max(phi))
    print(np.min(phi))
    print(np.max(psi))
    print(np.min(psi))

    gradient = np.zeros(image.shape)
    
    return phi, psi

def build_r_table(image, origin):
    '''
    Build the R-table from the given shape image and a reference point
    '''
    
    dx = sobel(image, 0)  # x derivative
    dy = sobel(image, 1)  # y derivative
    dz = sobel(image, 2)  # z derivative
    
    mag = np.sqrt(dx*dx + dy*dy + dz*dz)
    mag_norm = mag/np.max(mag)
    
    
    print("Magnitude: ",mag.shape)
    #mag = generic_gradient_magnitude(image, sobel)
        
    print(dx[0,0,0], dy[0,0,0], dz[0,0,0], mag_norm[0,0,0])
    
    print("dx,dy,dz: ", dx.shape,dy.shape,dz.shape)
    
    #Creating edge array the same size as query image
    edges = np.zeros(image.shape, dtype=bool)
    print("Edge: ", edges.shape)
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for k in range(edges.shape[2]):
                if mag_norm[i,j,k] > 0.3 :# and mag_norm[i,j,k] < 0.9:
                    edges[i,j,k] = True
    



    #Takes (47,40) Edges and calculates the gradients using sobel
    phi, psi = gradient_orientation(edges)
    print("Phi Dim: ", phi.shape)
    
    r_table = defaultdict(list)
    for (i,j,k),value in np.ndenumerate(edges):
        if value:
            r_table[(int(phi[i,j,k]),int(psi[i,j,k]))].append((origin[0]-i, origin[1]-j, origin[2] - k))
    
    print(r_table.keys())
    
    return r_table


def accumulate_gradients(r_table, grayImage):
    '''
    Perform a General Hough Transform with the given image and R-table
    '''
    #edges = canny(grayImage, low_threshold=MIN_CANNY_THRESHOLD, high_threshold=MAX_CANNY_THRESHOLD)
    
    dx = sobel(grayImage, 0)  # x derivative
    dy = sobel(grayImage, 1)  # y derivative
    dz = sobel(grayImage, 2)  # z derivative
    
    mag = np.sqrt(dx*dx + dy*dy + dz*dz)
    mag_norm = mag/np.max(mag)
    
    #Creating edge array the same size as query image
    edges = np.zeros(grayImage.shape, dtype=bool)
    #print("Edge: ", edges.shape)
    
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            for k in range(edges.shape[2]):
                if mag_norm[i,j,k] > 0.3 :# and mag_norm[i,j,k] < 0.9:
                    edges[i,j,k] = True      
    
    phi, psi = gradient_orientation(edges)
    
    accumulator = np.zeros(grayImage.shape)
    print("Accumulator shape: ", accumulator.shape)
    #print(accumulator)

    print("Start Accumulation")
    for (i,j,k),value in np.ndenumerate(edges):
        #print(i,j,k,value)
        if value:
            #print(r_table.keys())
            #Changed to int(gradient) which makes more sense
            for r in r_table[(phi[i,j,k], psi[i,j,k])]:
                accum_i, accum_j, accum_k = i+r[0], j+r[1], k+r[2]
                if accum_i < accumulator.shape[0] and accum_j < accumulator.shape[1] and accum_k < accumulator.shape[2]:
                    accumulator[int(accum_i), int(accum_j), int(accum_k)] += 1
        #print(i,j)
                    
    return accumulator

def general_hough_closure(reference_image):
    '''
    Generator function to create a closure with the reference image and origin
    at the center of the reference image
    
    Returns a function f, which takes a query image and returns the accumulator
    '''
    print(reference_image.shape)
    
    referencePoint = (reference_image.shape[0]/2, reference_image.shape[1]/2, reference_image.shape[2]/2)
    print(referencePoint)
    
    r_table = build_r_table(reference_image, referencePoint)
    
    def f(query_image):
        return accumulate_gradients(r_table, query_image)
        
        
    print("Finish General Hough Closure Function")

    return f

def n_max(a, n):
    '''
    Return the N max elements and indices in a
    '''
    indices = a.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, a.shape) for i in indices)
    return [(a[i], i) for i in indices]

def test_general_hough(gh, reference_image, query):
    '''
    Uses a GH closure to detect shapes in an image and create nice output
    '''
    #query_image = imread(query, flatten=True)
    query_image = query
    
    accumulator = gh(query_image)
    
    print("Accumulator Size: ", accumulator.shape)
    
    plt.clf() #Clear the Current Figure
    plt.gray() #Set colormap to gray
    
    
    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.title('Reference image')
    plt.imshow(reference_image[15,:,:])
    plt.show()
    fig.add_subplot(2,2,2)
    plt.title('Query image')
    plt.imshow(query_image[15,:,:])
    
    fig.add_subplot(2,2,3)
    plt.title('Accumulator')
    plt.imshow(accumulator[25,:,:])
    
    fig.add_subplot(2,2,4)
    plt.title('Detection')
    plt.imshow(query_image[25,:,:])
    
    # top 5 results in red
    m = n_max(accumulator, 10)
    
    print(m)
    
    
    x_points = [pt[1][0] for pt in m]
    y_points = [pt[1][1] for pt in m] 
    z_points = [pt[1][1] for pt  in m] 
    print(x_points)
    print(y_points)
    plt.scatter(y_points, z_points, marker='o', color='r')
    
    return


def test():
    #Testing with 3D Images
    np.random.seed(29)
    sample_3d = np.random.randint(0, 256, size=(50,50,50))

    print(np.shape(sample_3d))
    print(sample_3d[3,3,4])
    
    
    #Creating a test image
    test_3d = sample_3d
    

    
    detect_s = general_hough_closure(sample_3d)
    
    test_general_hough(detect_s, sample_3d, test_3d)


    

if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('C:\\Users\\yoons\\Documents\\4th Year Semester 1\\ESC499 - Thesis\\Generalized Hough Transform')
    test()