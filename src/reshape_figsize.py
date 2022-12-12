import cv2

def resize_fig(fig_num):
    height = 0
    width = 0

    for i in range(fig_num):
        path = '../data/fig/{}.jpg'.format(i)
        im = cv2.imread(path)
        height += im.shape[0]
        width += im.shape[1] 

    resize_h = height//fig_num
    resize_w = width//fig_num

    for i in range(fig_num):
        input_path = '../data/fig/{}.jpg'.format(i)
        output_path = '../data/reshape_fig/{}.jpg'.format(i)
        im = cv2.imread(input_path)
        im = cv2.resize(im, dsize = (resize_h, resize_w))
        cv2.imwrite(output_path, im)

if __name__ == '__main__':
    resize_fig(300)