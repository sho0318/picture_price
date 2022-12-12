import cv2

def resize_fig(fig_num):
    resize_h = 400
    resize_w = 400

    for i in range(fig_num):
        input_path = '../data/fig/{}.jpg'.format(i)
        output_path = '../data/reshape_fig/{}.jpg'.format(i)
        im = cv2.imread(input_path)
        im = cv2.resize(im, dsize = (resize_h, resize_w))
        cv2.imwrite(output_path, im)
    
    print("size to reshape = (", resize_h, "," ,resize_w, ")")

if __name__ == '__main__':
    resize_fig(300)