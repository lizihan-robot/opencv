import cv2 as cv
import numpy as np

def get_image_info(image):
    print(type(image))
    print(image.shape) #height,width,rgb
    print(image.size)
    print(image.dtype)
    pixel_data = np.array(image) #image is a numpy array
    # print(pixel_data)

def video_demo():
    capture = cv.VideoCapture(0) #open video
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame,1) #相机镜像 
        cv.imshow("video",frame)
        c = cv.waitKey(50)
        if c == 27:
            break
print("---------")

src = cv.imread("C:\\Users\\lizihan\\Pictures\\Saved Pictures\\Inkedd58ea351ba381297c1dfe6940f2dff5f_LI.jpg") # read image
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imwrite("C:\\Users\\lizihan\\Pictures\\Saved Pictures\\12.jpg",)


def access_pixels(image):
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("height: {},width: {}, channels:{}".format(height,width,channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255-pv #
    cv.imshow("pixels_demo",image)

def create_image():
    img = np.zeros([400, 400, 3],np.uint8)
    img[:,:,0] = np.ones([400,400])*255
    cv.imshow("new image",img)


# creat window show images
cv.namedWindow("input image",cv.WINDOW_NORMAL )
# cv.imshow("input image",src)
# access_pixels(src)
# get_image_info(src)
# video_demo()
create_image()
cv.waitKey(0)

cv.destroyAllWindows()