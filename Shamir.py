import time
import numpy as np
import argparse
import png
import sys
import os
from PIL import Image
from Crypto.Util.number import *

def preprocessing(path):
    img = Image.open(path)
    data = np.asarray(img)
    return data.flatten(),data.shape

def insert_text_chunk(src_png, dst_png, text):
    '''在png中的第二个chunk插入自定义内容'''
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()#创建一个每次返回一个chunk的生成器
    chunk_list = list(chunks)
    chunk_item = tuple([b'tEXt', text])

    index = 1
    chunk_list.insert(index, chunk_item)

    with open(dst_png, 'wb') as dst_file:
        png.write_chunks(dst_file, chunk_list)

def read_text_chunk(src_png, index=1):
    '''读取png的第index个chunk'''
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()
    chunk_list = list(chunks)
    img_extra = chunk_list[index][1].decode()
    img_extra = eval(img_extra)
    return img_extra

def polynomial(img, n, r):
    num_pixels = img.shape[0]
    # 生成多项式系数
    coefficients = np.random.randint(low = 0, high = 257, size = (num_pixels, r - 1))
    secret_imgs = []
    imgs_extra = []
    for i in range(1, n + 1):
        # 构造(r-1)次多项式
        base = np.array([i ** j for j in range(1, r)])
        base = np.matmul(coefficients, base)

        secret_img = (img + base) % 257

        indices = np.where(secret_img == 256)[0]
        img_extra = indices.tolist()
        secret_img[indices] = 0

        secret_imgs.append(secret_img)
        imgs_extra.append(img_extra)
    return np.array(secret_imgs), imgs_extra

def format_size(size_bytes):
    """ 根据字节大小自动调整单位 """
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def get_file_size(file_path):
    """ 获取文件大小并格式化输出 """
    try:
        size = os.path.getsize(file_path)
        return format_size(size)
    except OSError as e:
        return f"Error: {e}"

def lagrange(x, y, num_points, x_test):
    l = np.zeros(shape=(num_points,))
    for k in range(num_points):

        l[k] = 1
        for k_ in range(num_points):

            if k != k_:
                d = int(x[k] - x[k_])
                inv_d = inverse(d, 257)
                l[k] = l[k] * (x_test - x[k_]) * inv_d % 257

            else:
                pass
    L = 0
    for i in range(num_points):
        L += y[i] * l[i]
    return L


def decode(imgs,imgs_extra ,index, r):
    assert imgs.shape[0] >= r
    x = np.array(index)
    dim = imgs.shape[1]
    img = []

    print("decoding:")
    last_percent_reported = None
    imgs_add = np.zeros_like(imgs,dtype=np.int32)
    for i in range(r):
        for indices in imgs_extra[i]:
            imgs_add[i][indices] = 256

    for i in range(dim):
        y = imgs[:, i]
        ex_y = imgs_add[:, i]
        y = y + ex_y
        pixel = lagrange(x, y, r, 0) % 257
        img.append(pixel)

        # 计算当前进度
        percent_done = (i + 1) * 100 // dim
        if last_percent_reported != percent_done:
            if percent_done % 1 == 0:
                last_percent_reported = percent_done
                bar_length = 50
                block = int(bar_length * percent_done / 100)
                text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
                sys.stdout.write(text)
                sys.stdout.flush()

    print()

    return np.array(img)

def compare_images(image1_path, image2_path):
    image1 = np.array(Image.open(image1_path))
    image2 = np.array(Image.open(image2_path))
    diff = np.abs(image1 - image2)
    diff_value = round(np.mean(diff), 4)
    print("Mean difference:", diff_value)
    print("Max difference:", round(np.max(diff), 4))
    print("Min difference:", round(np.min(diff), 4))
    print("Standard deviation of difference:", round(np.std(diff), 4))


def main():
    parser = argparse.ArgumentParser(description='Shamir Secret Image Sharing')
    parser.add_argument('-e', '--encode', help='Path to the image to be encoded')
    parser.add_argument('-d', '--decode', help='Path for the origin image to be saved')
    parser.add_argument('-n', type=int, help='The total number of shares')
    parser.add_argument('-r', type=int, help='The threshold number of shares to reconstruct the image')
    parser.add_argument('-i', '--index', nargs='+', type=int, help='The index of shares to use for decoding')
    parser.add_argument('-c', '--compare', nargs=2, help='Compare two images')
    args = parser.parse_args()

    if args.encode:
        start_time = time.time()
        print("\n=== Starting image encoding process ===")

        if not args.r:
            print("Error: Threshold number 'r' is required for decoding")
            return
        if not args.n:
            print("Error: Total number 'n' of shares is required for decoding")
            return
        if args.r > args.n:
            print("Error: Threshold 'r' cannot be greater than the total number 'n' of shares")
            return

        img_flattened, shape = preprocessing(args.encode)
        secret_imgs, imgs_extra = polynomial(img_flattened, n=args.n, r=args.r)
        to_save = secret_imgs.reshape(args.n, *shape)
        for i, img in enumerate(to_save):
            secret_img_path = f"secret_{i + 1}.png"
            Image.fromarray(img.astype(np.uint8)).save(secret_img_path)
            img_extra = str(list((imgs_extra[i]))).encode()
            insert_text_chunk(secret_img_path, secret_img_path, img_extra)
            size = get_file_size(secret_img_path)
            print(f"{secret_img_path} saved.",size)
            

        end_time = time.time()
        print("=== Image encoding completed. Time elapsed: {:.2f} seconds ===".format(end_time - start_time))

    if args.decode:
        start_time = time.time()
        print("\n=== Starting image decoding process ===")

        if not args.r:
            print("Error: Threshold number 'r' is required for decoding")
            return

        input_imgs = []
        input_imgs_extra = []
        for i in args.index:
            secret_img_path = f"secret_{i}.png"
            img_extra = read_text_chunk(secret_img_path)
            img, shape = preprocessing(secret_img_path)
            input_imgs.append(img)
            input_imgs_extra.append(img_extra)
        input_imgs = np.array(input_imgs)
        origin_img = decode(input_imgs, input_imgs_extra, args.index, r=args.r)
        origin_img = origin_img.reshape(*shape)
        Image.fromarray(origin_img.astype(np.uint8)).save(args.decode)
        size = get_file_size(args.decode)
        print(f"{args.decode} saved.",size)

        end_time = time.time()
        print("=== Image decoding completed. Time elapsed: {:.2f} seconds ===".format(end_time - start_time))

    if args.compare:
        print("\n=== Starting image comparison ===")

        compare_images(args.compare[0], args.compare[1])

        print("=== Image comparison completed.  ===")

if __name__ == "__main__":
    main()
