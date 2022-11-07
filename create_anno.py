import numpy as np
import cv2
import os
import json
import math

COLOR_PolarRGB = "COLOR_PolarRGB"
COLOR_PolarMono = "COLOR_PolarMono"


def Aop_transfer(image):
    pi_1_t = np.where(image >= math.pi / 2, 1, 0)
    pi_1 = np.multiply(pi_1_t, image)
    pi_2_t = np.where(image < math.pi / 2, 1, 0)
    pi_2 = np.multiply(pi_2_t, image)

    array2 = np.float64(np.full((2056, 2464), 0.1))

    R_1 = np.log10(np.multiply(10, np.divide(np.sin(pi_1), 1.1) + array2))
    R_2 = np.multiply(1 - np.cos(pi_2), np.log10(np.multiply(10, np.divide(np.sin(pi_2), 1.1) + array2)))
    R_1 = np.multiply(pi_1_t, R_1)
    R_2 = np.multiply(pi_2_t, R_2)

    R = R_1 + R_2

    G_1 = np.multiply(1 + np.cos(pi_1), np.log10(np.multiply(10, np.divide(np.sin(pi_1), 1.1) + array2)))
    G_2 = R_2
    G_1 = np.multiply(pi_1_t, G_1)

    G = G_1 + G_2

    B_1 = G_1
    B_2 = np.log10(np.multiply(10, np.divide(np.sin(pi_2), 1.1) + array2))
    B_1 = np.multiply(pi_1_t, B_1)

    B = B_1 + B_2
    images = np.zeros((2056, 2464,3))
    images[:, :, 0] = B
    images[:, :, 1] = G
    images[:, :, 2] = R
    return images


def rotator(theta):
    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    sin2 = np.sin(2*theta)
    cos2 = np.cos(2*theta)
    mueller = np.array([[ones,  zeros, zeros, zeros],
                  [zeros,  cos2,  sin2, zeros],
                  [zeros, -sin2,  cos2, zeros],
                  [zeros, zeros, zeros, ones]])
    mueller = np.moveaxis(mueller, [0,1], [-2,-1])
    return mueller


def rotateMueller(mueller, theta):

    return rotator(-theta) @ mueller @ rotator(theta)

def polarizer(theta):
    mueller = np.array([[0.5, 0.5, 0, 0],
                  [0.5, 0.5, 0, 0],
                  [  0,   0, 0, 0],
                  [  0,   0, 0, 0]]) # (4, 4)
    mueller = rotateMueller(mueller, theta)
    return mueller

def calcLinearStokes(intensities, thetas):
    muellers = [polarizer(theta)[..., :3, :3] for theta in thetas]
    return calcStokes(intensities, muellers)

def applyColorToAoLP(img_AoLP, saturation=1.0, value=1.0):
    img_ones = np.ones_like(img_AoLP)

    img_hue = (np.mod(img_AoLP, np.pi) / np.pi * 140).astype(np.uint8)  # 0~pi -> 0~179
    img_saturation = np.clip(img_ones * saturation * 255, 0, 255).astype(np.uint8)
    img_value = np.clip(img_ones * value * 255, 0, 255).astype(np.uint8)

    img_hsv = cv2.merge([img_hue, img_saturation, img_value])
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_bgr

def cvtStokesToAoLP(img_stokes):

    S1 = img_stokes[..., 1]
    S2 = img_stokes[..., 2]
    return np.mod(0.5*np.arctan2(S2, S1), np.pi)


def calcStokes(intensities, muellers):

    if not isinstance(intensities, np.ndarray):
        intensities = np.stack(intensities, axis=-1) # (height, width, n)

    if not isinstance(muellers, np.ndarray):
        muellers = np.stack(muellers, axis=-1) # (3, 3, n) or (4, 4, n)

    if muellers.ndim == 1:
        # 1D array case
        thetas = muellers
        return calcLinearStokes(intensities, thetas)

    A = muellers[0].T # [m11, m12, m13] (n, 3) or [m11, m12, m13, m14] (n, 4)
    A_pinv = np.linalg.pinv(A) # (3, n)
    stokes = np.tensordot(A_pinv, intensities, axes=(1, -1)) # (3, height, width) or (4, height, width)
    stokes = np.moveaxis(stokes, 0, -1) # (height, width, 3)



    return stokes


def demosaicing(img_raw, code=COLOR_PolarMono):
    if code == COLOR_PolarMono:
        if img_raw.dtype == np.uint8 or img_raw.dtype == np.uint16:
            return __demosaicing_mono_uint(img_raw)
        else:
            return __demosaicing_mono_float(img_raw)
    elif code == COLOR_PolarRGB:
        if img_raw.dtype == np.uint8 or img_raw.dtype == np.uint16:
            return __demosaicing_color(img_raw)
        else:
            raise TypeError("dtype of `img_raw` must be np.uint8 or np.uint16")
    else:
        raise ValueError(f"`code` must be {COLOR_PolarMono} or {COLOR_PolarRGB}")


def __demosaicing_mono_uint(img_mpfa):

    img_debayer_bg = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerBG2BGR)
    img_debayer_gr = cv2.cvtColor(img_mpfa, cv2.COLOR_BayerGR2BGR)
    img_0 = img_debayer_bg[:, :, 0]
    img_90 = img_debayer_bg[:, :, 2]
    img_45 = img_debayer_gr[:, :, 0]
    img_135 = img_debayer_gr[:, :, 2]
    img_polarization = np.array([img_0, img_45, img_90, img_135], dtype=img_mpfa.dtype)
    img_polarization = np.moveaxis(img_polarization, 0, -1)
    return img_polarization


def __demosaicing_mono_float(img_mpfa):
    height, width = img_mpfa.shape[:2]
    img_subsampled = np.zeros((height, width, 4), dtype=img_mpfa.dtype)

    img_subsampled[0::2, 0::2, 0] = img_mpfa[0::2, 0::2]
    img_subsampled[0::2, 1::2, 1] = img_mpfa[0::2, 1::2]
    img_subsampled[1::2, 0::2, 2] = img_mpfa[1::2, 0::2]
    img_subsampled[1::2, 1::2, 3] = img_mpfa[1::2, 1::2]

    kernel = np.array([[1 / 4, 1 / 2, 1 / 4],
                       [1 / 2, 1.0, 1 / 2],
                       [1 / 4, 1 / 2, 1 / 4]])

    img_polarization = cv2.filter2D(img_subsampled, -1, kernel)

    return img_polarization[..., [3, 1, 0, 2]]


def __demosaicing_color(img_cpfa):

    height, width = img_cpfa.shape[:2]

    # 1. Color demosaicing process
    img_mpfa_bgr = np.empty((height, width, 3), dtype=img_cpfa.dtype)
    for j in range(2):
        for i in range(2):
            # (i, j)
            # (0, 0) is 90,  (0, 1) is 45
            # (1, 0) is 135, (1, 1) is 0

            # Down sampling ↓2
            img_bayer_ij = img_cpfa[j::2, i::2]
            # Color demosaicking
            img_bgr_ij = cv2.cvtColor(img_bayer_ij, cv2.COLOR_BayerBG2BGR)
            # Up samping ↑2
            img_mpfa_bgr[j::2, i::2] = img_bgr_ij

    # 2. Polarization demosaicing process
    img_bgr_polarization = np.empty((height, width, 3, 4), dtype=img_mpfa_bgr.dtype)
    for i, img_mpfa in enumerate(cv2.split(img_mpfa_bgr)):
        img_demosaiced = demosaicing(img_mpfa, COLOR_PolarMono)
        img_bgr_polarization[..., i, :] = img_demosaiced

    return img_bgr_polarization



def genmaps(H, W, plist):
    line = np.zeros((H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)
    for i in range(0, H, 10):
        cv2.line(line, (0, i), (W, i), (255), 1)
    
    for i in range(len(plist)-1):
        cv2.line(mask, (int(plist[i][0]), int(plist[i][1])), 
                        (int(plist[i+1][0]), int(plist[i+1][1]) ), (255), 1)
    
    return mask.astype(int), line.astype(int)

def create_anno(path, directory, list_c):


    anno_order = dict()


    for i,data in enumerate(directory):
        write_dir = str(i)
        anno_order[data] = write_dir
        for img in list_c[data]:
            dir_name = path+'/'+write_dir
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            img_t = cv2.imread(img[1])

            img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)
            # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            p_img = demosaicing(img_t)

            radians = np.array([0, np.pi / 4, np.pi / 2, np.pi * 3 / 4])
            img_stokes = calcStokes(p_img, radians)
            img_AoLP = cvtStokesToAoLP(img_stokes)
            
            img_AoLP_color = applyColorToAoLP(img_AoLP)
            img_stokes = cv2.resize(img_AoLP_color, dsize=(1280, 720))
            # img_stokes = np.uint8(np.multiply(img_stokes,255))
            cv2.imwrite(dir_name + '/'+img[0][:-3]+'jpg',img_stokes)

        list_c[data]=0






    # des_dir = 'anno'
    # hs = 10
    # json_f = open('test.json', 'w')
    # for path in list_c['anno']:
    #     write_dir = 'test/'+str(anno_order[path.split('/')[6]])
    #     if not path.endswith('json'): continue
    #     with open(os.path.join(des_dir, path), 'r') as f:
    #         data = json.load(f)
    #     H, W = data['imageHeight'], data['imageWidth']
    #     lanes = []
    #     h_sample = np.arange(360, 720, 10)
    #     for dd in data['shapes']:
    #         plist = np.array(dd['points'])
    #         mask, line = genmaps(H, W, plist)
    #         ml = mask & line
    #         rows, cols = np.nonzero(ml)
    #         rows = np.uint64(rows*720/2056)
    #         cols = np.uint64(cols*1280/2464)
    #
    #         lane = np.ones_like(h_sample) * -2
    #         for i, value in enumerate(rows):
    #
    #             if value//10>35:
    #                 lane[int(value//10)-36] = cols[i]
    #         lanes.append(lane.tolist())
    #     info = {}
    #     info["lanes"] = lanes
    #
    #     info["raw_file"] = os.path.join(write_dir, path.split('/')[7].split('.')[0]+'.jpg')
    #     info["h_samples"] = h_sample.tolist()
    #     json_f.write(json.dumps(info))
    #     json_f.write('\n')
    #
    # json_f.close()





def Load_Data(path):

    directory =[]
    list_img = dict()
    if os.path.isdir(path):
        data = os.listdir(path)

        for some in data:

            directory.append(some)
        json = []
        for img_dir in directory:
            l_path = path +'/' + img_dir
            img_list = os.listdir(l_path)
            imgs =[]

            for img in img_list:
                if img[-3:]=='png'or img[-3:]=='jpg':
                    img_path = l_path+'/'+img
                    #data_img = cv2.imread(img_path)
                    img_info = []
                    img_info.append(img)
                    img_info.append(img_path)
                    imgs.append(img_info)
                else:
                    json.append(path +'/'+img_dir+'/'+img)

            list_img[img_dir] = imgs
        list_img['anno'] = json



    return directory , list_img



if __name__ == '__main__':
    path = '/home/a/Downloads/lane_dataset/test'
    directory, list_c = Load_Data(path)
    create_anno(path, directory, list_c)
    print("end")






