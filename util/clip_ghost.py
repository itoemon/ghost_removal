import cv2

drawing = False
ix, iy, width_diff, height_diff = 30, 30, 0, 0
width = 500
height = 500


def clip_ghost(img, wid, hei):


    global ix, iy, drawing, width_diff, height_diff, width, height

    width = wid
    height = hei

    temp = img.copy()

    source_window = "push s for save, esc for quite"
    cv2.namedWindow(source_window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(source_window, my_mouse_callback)

    while True:
        cv2.imshow(source_window,temp)
        cv2.rectangle(temp, (ix, iy), (ix + width, iy + height), (0,255,0), 2)  # 矩形を描画

        if(drawing):            # 左クリック押されてたら
            temp = img.copy()   # 画像コピー
            cv2.rectangle(temp, (ix, iy), (ix + width, iy + height), (0,255,0), 2)  # 矩形を描画

        # キー操作
        k = cv2.waitKey(1) & 0xFF
        if k == 27 and not drawing:             # esc押されたら終了
            break

        elif k == ord('s') and not drawing:
            img_height, img_width = img.shape[0:2]

            if (ix >=0 and ix + width <= img_width) and (iy >= 0 and iy + height <= img_height):
                result = img[iy:iy + height, ix:ix + width, :]
                top = iy
                bottom = img_height - (iy + height)
                left = ix
                right = img_width - (ix + width)
                result_pad = cv2.copyMakeBorder(result, top, bottom, left, right, cv2.BORDER_CONSTANT, (0,0,0))
                result2 = img - result_pad

                return result, result2, top, bottom, left, right
            
    cv2.destroyAllWindows()


# マウスコールバック関数
def my_mouse_callback(event, x, y, flags,param):

    global ix, iy, drawing, width_diff, height_diff
 
    if event == cv2.EVENT_MOUSEMOVE:      # マウスが動いた時
        if(drawing == True):
            ix = x - width_diff
            iy = y - height_diff
 
    elif event == cv2.EVENT_LBUTTONDOWN:  # マウス左押された時
        if (ix < x < ix + width) and (iy < y < iy + height):        
            width_diff = x - ix
            height_diff = y - iy
            drawing = True
 
    elif event == cv2.EVENT_LBUTTONUP:    # マウス左離された時
        drawing = False


if __name__ == "__main__":
    fname = '../data/test/test (1).png'
    img = cv2.imread(fname)
    result, result2, top, bottom, left, right = clip_ghost(img)

    cv2.imwrite('result.png', result)
    cv2.imwrite('result2.png', result2)