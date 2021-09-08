import matplotlib
import numpy as np 
from scipy.ndimage import uniform_filter

def extract_features(imgs, feature_fns, verbose = False):
    """
    输入图像的像素信息，利用特征函数对图像进行特征提取

    输入：
    - imgs：N 张图像的像素数组，大小为 N x H x W x C
    - feature_fns:  有 k 个特征函数的列表
    - verbose：如果为 True，打印过程
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # 使用第一张图像确定特征维度
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature funtions mumt be one-dimentional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)
    
    # 既然我们已经确定了特征的维度，那么我们就可以分配一个大数组用来存储所有的特征，每个特征作为一列
    total_features_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_features_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # 对剩下的图像提取特征
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print('Done extracting featurs for %d / %d images' % (i, num_images))
    
    return imgs_features

def rgb2gray(rgb):
    """
    将 RGB 图像转换到灰度级
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def hog_feature(im):
    """
    计算图像的梯度（HOG）特征直方图
    """
    # 将 rgb 转换为灰度级
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)
    
    sx, sy = image.shape
    orientations = 9 # 梯度箱的数量
    cx, cy = (8, 8) # 每个单元的像素

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n = 1, axis = 1) # 计算 x 方向上的梯度
    gy[:-1, :] = np.diff(image, n = 1, axis = 0) # 计算 y 方向上的梯度
    grad_mag = np.sqrt(gx ** 2 + gy ** 2) # 梯度幅度
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90 # 梯度方向

    n_cellsx = int(np.floor(sx / cx)) # 在 x 方向上的单元数
    n_cellsy = int(np.floor(sy / cy)) # 在 y 方向上的单元数
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                        grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                        temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size = (cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T

    return orientation_histogram.ravel()

def color_histogram_hsv(im, nbin = 10, xmin = 0, xmax = 255, normalized = True):
        """
        使用色调计算图像的颜色直方图

        输入：
        - im：一张 RGB 图像的像素信息，大小为 H x W x C
        - nbin：直方图箱数
        - xmin：最小像素值
        - xmax：最大像素值
        - normalized：是否对直方图进行规范化

        返回：
        一个一维的向量，长度为 nbin，存储输入图像的颜色直方图信息
        """
        ndim = im.ndim
        bins = np.linspace(xmin, xmax, nbin + 1)
        hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
        imhist, bin_edges = np.histogram(hsv[:, :, 0], bins = bins, density = normalized)
        imhist = imhist * np.diff(bin_edges)

        return imhist