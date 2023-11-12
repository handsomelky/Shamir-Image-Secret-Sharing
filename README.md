# Shamir's Secret Sharing for Images

This repository implements a secret sharing algorithm for images based on Shamir's Secret Sharing scheme, which enables the encryption and decryption of images into multiple shares, ensuring the original image can be losslessly reconstructed from a subset of the shares.

[简体中文](readme/README.zh-CN.md) | [繁體中文](readme/README.zh-TW.md) | English

## Features

- Secure encryption and decryption of images using the Shamir scheme.
- Supports both grayscale and color images.
- Zero loss of pixel data (lossless recovery), including edge cases with pixel values of 256.
- Efficient metadata storage for necessary reconstruction information.
- Detailed progress indicators and performance metrics display.

## Installation

To use this secret sharing algorithm, you need to install Python and several dependencies on your system.

- Python 3.x
- NumPy
- Pillow
- PyCryptodome
- pypng

You can install all required packages with the following command:

```shell
pip install -r requirements.txt
```

## How to Use

To encrypt (share) an image:

```shell
python Shamir.py -e <image-path> -n <number-of-shares> -r <threshold>
```

To decrypt (reconstruct) an image using shares:

```shell
python Shamir.py -d <output-path> -r <threshold> -i <share-indexes>
```

To compare the original and reconstructed images:

```shell
python Shamir.py -c <original-image-path> <reconstructed-image-path>
```

For a full list of options, use the `-h` flag:

```shell
python Shamir.py -h
```

## Example

Here is an example of how to encode and decode an image:

```shell
# Encode the image into 5 shares with a threshold of 3
python Shamir.py -e avatar.png -n 5 -r 3

# Decode the image using shares 1, 4, and 5
python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5

# Compare the original and recovered images
python Shamir.py -c avatar.png avatar_recover.png
```

## Design Philosophy

The full design philosophy has been uploaded to my blog:[面向图像的秘密共享算法设计 | R1ck's Portal (rickliu.com)](https://rickliu.com/posts/0742023ea0bf/)

In reality, when we need to share image-type information secretly, the secret sharing scheme for character data still works well.

Since every pixel of an image can be represented by RGB values (or a grayscale value), we can find a suitable scheme to encrypt and share the color values, simply by adding some **preprocessing** and **postprocessing** to the original secret sharing scheme.

Preprocessing involves extracting image features and converting them into binary data.

Postprocessing, then, is about turning the decrypted binary data back into an image.

And the intermediate form of the encrypted data can also be an image, albeit appearing as a jumble of noise.

### Algorithm Selection

The basic algorithmic framework we chose is the **Shamir Secret Sharing Scheme**.

Compared to the power operations in CRT, Shamir's **polynomial operations are more suited for computing across the entire image array at once**.

But there is one point to note in Shamir's secret sharing for images:

$$S = F(0) = \sum_{j=1}^{T} F(x_j) \prod_{\substack{l=1 \\ l \neq j}}^{T} \frac{x_l}{x_l - x_j} \mod q$$

Here, the modulus q needs to be a prime number, and the bit depth of each channel of our shadow images is only 8 bits, which is 0 to 255.

If the modulus q we choose is less than 255, then the precision of some pixel points calculated polynomial values will be lost.

And if the modulus we choose is greater than 255, then **if the polynomial value of a pixel point after modding 257 is greater than 255, we cannot directly save it as a pixel value**, and this part of the pixel point data cannot be transmitted directly.

So the key question is how to save the extra information caused by the modulus.

Here we choose a modulus close to 256, which is 257, so we only need to record one situation, that is, the polynomial value of mod 257 is 256.

**We consider putting the indices of these 256 pixels in the image array into a list and saving this list in the image metadata (Meta data)**.

### Shadow Image Generation

The steps for generating a shadow image are shown in the following figure:

![Generating secret images](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E7%94%9F%E6%88%90%E7%A7%98%E5%AF%86%E5%9B%BE%E7%89%87-169971971748022.png)

1. Randomly generate polynomial coefficients, with the original pixel value (the secret) as $a_0$.
2. Flatten the original image array so that the values on the three channels R, G, and B can be computed polynomially at the same time, and the results are combined into a color shadow image.
3. If the value of the pixel on the shadow image generated is 256, set it to 0 and save the index information in the metadata
4. Traverse the indices until N shadow images are generated

### Original Image Recovery

To recover the original image, you need to first restore the index array stored in the metadata to an image array and add it directly to the array of the secret image, thus **recovering the information of the pixels with a remainder of 256**.

The recovery algorithm then uses the **Lagrange interpolation method**.

The steps for recovering the original image are shown in the following figure:

![Recovery of the original image](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E5%8E%9F%E5%9B%BE%E5%83%8F%E6%81%A2%E5%A4%8D-169971971748023.png)

1. Read and process r shadow images, each image's three-channel values are flattened into a one-dimensional array
2. Extract and save the extra information from the metadata of the shadow image, restore it to a one-dimensional array composed of 256 pixels, and **directly add** it to the array of the shadow image
3. Each time one pixel value is read from the array of r shadow images, use the Lagrange interpolation method to recover one pixel value of the original image

### Innovations

At first, I considered transforming the program into a multi-threaded one and making it an innovation point. But then I realized, **secret sharing is actually a computation-intensive process**. For Python's concurrent programming, I/O-intensive programs can make better use of multi-threading than computation-intensive programs. **Using multi-threading in this task does not significantly improve speed**.

#### Lossless Recovery

The **biggest innovation** of this algorithm is that it **realizes a lossless image secret sharing process, and the size of the generated shadow image is not too large**.

The more common mods 251 and 257 will cause loss of original image information to varying degrees.

If we can save the mod 256 information additionally in the image metadata, we can achieve lossless recovery.

Selecting the png format for the intermediate shadow image, we can quickly store the extra information in the file's chunk.

```python
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
```

So the extra information to be stored in the metadata needs to ensure a **very high information density**, and it is obvious that storing **the indices of the 256 remains in an array** is an appropriate choice.

Through the **where() method** provided by numpy, we can quickly find the indices of elements with a value of 256 in the array and use these indices to set to 0.

```python
indices = np.where(secret_img == 256)[0]
img_extra = indices.tolist()
secret_img[indices] = 0
```

During image recovery, we only need to add the array of the shadow image to the array of the same size recovered from the extra information:

```python
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
```

#### All Functions in a Single Python File

Another innovation point is that the entire secret sharing process is integrated into a single Python source code. When running this Python file from the console, **by setting options and passing in different parameters**, we can complete three tasks: generating shadow images, recovering the original image, and comparing image pixel values.

This one-stop solution greatly simplifies the operation process, and users do not need to switch between different programs or scripts to complete the entire secret sharing cycle.

Key options in the program are explained as follows:

- `-e` / `--encode`: This option is followed by the path of the original image to specify the image file to be encrypted for secret sharing.
- `-d` / `--decode`: This option is followed by the save path of the decrypted image to specify the output directory of the decryption operation.
- `-n`: The parameter following this option sets the total number of shadow images to be generated, that is, the number of pieces of secret sharing.
- `-r`: The parameter following this option sets the minimum number of shadow images required to reconstruct the original image, that is, the threshold for secret sharing.
- `-i` / `--index`: This option accepts one or more integer parameters, representing the index of the shadow images used for decryption.
- `-c` / `--compare`: This option is followed by the paths of two image files to compare the differences between these two images.

In addition, you can use the `-h` parameter to bring up the program's manual, **all of which are thanks to Python's argparse library**.

It is worth noting that when decrypting, the shadow images need to be stored in the same path as Shamir.py, named according to the rule of secret_{index}, and in PNG format.

#### Intuitive and Detailed Display

With a carefully designed command-line interface, this program provides clear progress feedback and detailed status information when performing various operations, such as encrypting, decrypting, and comparing images. For example, during the decryption process, the program will not only **display the current progress bar** but will also output the **total time used for decryption** after completion, allowing users to clearly understand the efficiency of task execution.

The specific implementation is as follows:

1. **Progress Bar Display**: In the `decode` function, by calculating the proportion of the currently processed pixels to the total number of pixels, we have implemented a dynamically updated progress bar. This progress bar not only provides immediate feedback on the decryption process visually but also accurately expresses the current completion status in percentages.

   ```python
   percent_done = (i + 1) * 100 // dim
   if last_percent_reported != percent_done:
       if percent_done % 1 == 0:  # 每增加1%更新一次进度
           last_percent_reported = percent_done
           bar_length = 50
           block = int(bar_length * percent_done / 100)
           text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
           sys.stdout.write(text)
           sys.stdout.flush()
   ```

2. **Dynamic Display of File Size**: Using the `get_file_size` function, the program outputs the size of each shadow image and the recovered original image after saving. This size is dynamically calculated and formatted, automatically selecting the most appropriate unit (such as B, KB, MB) based on the actual size of the file, making the information display more intuitive.

   ```python
   size = get_file_size(secret_img_path)
   print(f"{secret_img_path} saved.", size)
   ```

3. **Detailed Image Comparison Report**: When comparing two images, the `compare_images` function not only outputs the **average difference value between the two images but also outputs the maximum difference, minimum difference, and the standard deviation of the difference**, providing users with a comprehensive image difference analysis. The **results of this report fully demonstrate the reliability of this algorithm in lossless secret sharing**.

   ```python
   print("Mean difference:", diff_value)
   print("Max difference:", round(np.max(diff), 4))
   print("Min difference:", round(np.min(diff), 4))
   print("Standard deviation of difference:", round(np.std(diff), 4))
   ```

4. **Intuitive Display of Running Time**: At key points in the program, such as after encryption or decryption is completed, the program will calculate and display the total time spent on the operation. This not only provides immediate feedback on the operation but also allows users to assess the performance of the program. By recording timestamps at the start and end of the operation, the program can output the running time accurate to milliseconds, making the performance test results more accurate.

   ```python
   start_time = time.time()  # 操作开始前记录时间
   # ... 执行操作 ...
   end_time = time.time()    # 操作结束后记录时间
   print(f"Operation completed. Time elapsed: {end_time - start_time:.2f} seconds.")
   ```

The implementation of the above features ensures that users get detailed operation information when using this program, including operation progress, file size, and operation time. These intuitive display information not only improve the transparency of user operations but also enhance users' confidence in program performance.

## Experimental Results

In the experiment, I will demonstrate the **execution effects of different functions of the program** and test the improvement of the lossless module implemented by this program in various aspects through **ablation experiments**.

I will use my avatar as a test sample, and the original image avatar.png is as follows:

![test image](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/test-169971971748124.jpg)

The original image size is 243KB, and the dimensions are 640*640.

The reason for choosing the PNG format for the original image is that PNG is a lossless compression image format, which means that the pixel data will not change when the image is recovered, which is more conducive to our precise testing of whether the entire process is a lossless secret sharing.

### Function Testing

First, we generate shadow images by executing the following command:

```shell
python Shamir.py -e avatar.png -n 5 -r 3
```

Successfully generated 5 shadow images.

![image-20231111212119557](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212119557-169971971748125.png)

![image-20231111212211000](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212211000-169971971748126.png)

Next, we use the image recovery function, executing the following command:

```shell
python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5
```

We choose the shadow images numbered 1, 4, and 5 to recover the original image.

![image-20231111212253826](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212253826-169971971748127.png)

![image-20231111212319416](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212319416-169971971748128.png)

Finally, we compare the pixel value differences between the recovered image and the original image.

Execute the following command:

```shell
python Shamir.py -c avatar_recover.png avatar.png
```

![image-20231111212340635](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212340635-1699720606808129.png)

**The restored image is completely identical to the original image**, indicating the successful realization of lossless image secret sharing.

If we want to execute all tasks simultaneously, including encryption, decryption, and comparison, we can use the following command:

``` shell
python Shamir.py -e avatar.png -n 5 -r 3 -d avatar_recover.png -i 1 4 5 -c avatar_recover.png avatar.png
```

![Command Execution for All Tasks](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111213341368-1699720646019132.png)

### Ablation Experiment

In this ablation experiment, we examine the impact of the "extra information" module.

By removing the content related to the extra section from the source code, we re-executed the entire process.

The image below shows the algorithm's results and the restored image **after removing the extra information module**:

![Algorithm Results without Extra Information Module](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111213949707-169972058927693.png)

![Restored Image without Extra Information Module](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/avatar_recover-16997100367043-169972058927794.png)

The next image shows the algorithm's results and the restored image **with the extra information module**:

![Algorithm Results with Extra Information Module](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111213701330-169972058927795.png)

![Restored Image with Extra Information Module](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/avatar_recover-169972058927796.png)

Firstly, regarding losslessness, the algorithm without the extra information module has an average pixel value difference of 0.9862 due to the effect of remainder 256 pixels. In contrast, the algorithm with the extra information module restores the image perfectly, with an average pixel value difference of 0.

![Pixel Value Difference without Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111214529421-169972058927797.png)

![Pixel Value Difference with Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111214822083-169972058927798.png)

Direct observation of the restored images provides a more intuitive understanding of the reasons behind these differences.

![Restored Image Observation without Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111214852488-169972058927799.png)

![Restored Image Observation with Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111220014444-1699720589277100.png)

The algorithm without the extra information module shows many pixels that failed to recover due to lost information.

These differences indicate that **the extra information module ensures all information is preserved, achieving lossless secret sharing**.

Beyond image directives, we should also pay attention to the algorithm's time and the size of the shadow images.

More calculations and storage are performed in the extra information module; we need to understand its impact on user experience.

![Algorithm Time without Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215428807-1699720589277101.png)

![Algorithm Time with Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215414634-1699720589277102.png)

It's evident that the algorithm's runtime changes minimally with the addition of the extra information module, indicating that **the efficiency of adding information in metadata is very high, and the efficiency of recovering the remainder 256 points during decoding is also high**.

![Shadow Image Size without Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215701758-1699720589277103.png)

![Shadow Image Size with Extra Information](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215723298-1699720589277104.png)

The information added in the metadata accounts for about 3% of the shadow image size, suggesting that the extra information added is **highly dense, ensuring lossless secret sharing while not consuming more space resources**.

## Contributions

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.

## License 

This project is licensed under the GNU General Public License v3.0 

see the [LICENSE](LICENSE) file for details. 