# 圖像秘密共享 - Shamir's Secret Sharing

本項目實現了基於Shamir秘密共享方案的圖像秘密共享算法，用於將圖像加密和解密為多個份額，並確保原始圖像可以從一部分份額中無損地重構。

[English](../README.md) | [简体中文](README.zh-CN.md) | 繁體中文

## 功能特性

- 使用Shamir方案安全加密和解密圖像。
- 支持灰度和彩色圖像。
- 像素數據零損失（無損恢復），包括像素值為256的邊緣情況。
- 高效的元數據存儲必要的恢復重建信息。
- 詳細的進度指示器和性能指標顯示。

## 安裝

要使用此秘密共享算法，需要在系統上安裝Python和幾個依賴庫。

- Python 3.x
- NumPy
- Pillow
- PyCryptodome
- pypng

您可以使用以下命令安裝所有所需的包：

```shell
pip install -r requirements
```

## 使用方法

加密（分享）圖像：

```shell
python Shamir.py -e <圖片路徑> -n <份額數> -r <閾值>
```

使用份額解密（重構）圖像：

```shell
python Shamir.py -d <輸出路徑> -r <閾值> -i <份額索引>
```

比較原始圖像和重構圖像：

```shell
python Shamir.py -c <原始圖像路徑> <重構圖像路徑>
```

獲取完整的選項列表，請使用`-h`標誌：

```shell
python Shamir.py -h
```

## 示例

以下是如何編碼和解碼圖像的示例：

```shell
# 將圖像編碼為5個份額，閾值為3
python Shamir.py -e avatar.png -n 5 -r 3

# 使用份額1、4和5解碼圖像
python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5

# 比較原始圖像和恢復圖像
python Shamir.py -c avatar.png avatar_recover.png
```



## 設計思路

我已經將完整的設計思路上傳到了博客中：[面向圖像的秘密共享算法設計 | R1ck's Portal (rickliu.com)](https://rickliu.com/posts/0742023ea0bf/)

事實上，當我們需要對圖像類型的信息進行秘密共享時，針對字符數據的秘密共享方案依然能夠很好的工作

因為圖像的每個像素點都可以由RGB表示（灰度圖為一個灰度值），所以我們可以找到合適的方案對色值進行加密並共享，只需在原先的秘密共享方案上增加一些**預處理**和**後處理**的過程

預處理即將圖像特征提取，轉化為二進製數據

而後處理則是將解密後的二進製數據轉回圖像

而密文數據的中間傳輸形態同樣也可以是圖像，雖然看起來是一些雜亂無章的噪點

### 算法選擇

基本的算法框架我們選用**Shamir秘密共享方案**

相比於CRT中進行的冪運算，Shamir中涉及的**多項式運算更適合同時對整個圖像數組進行計算**

但是在圖像類型的Shamir秘密共享中**有一點需要註意**：

$$S = F(0) = \sum_{j=1}^{T} F(x_j) \prod_{\substack{l=1 \\ l \neq j}}^{T} \frac{x_l}{x_l - x_j} \mod q$$

這裏的模數q需要選取一個素數，而我們保存的影子圖像的每個通道位深只有8位，即0到255

如果我們選擇的模數q如果小於255，那麽一些像素點算出來的多項式值的精度便會丟失

而如果我們選擇的模數大於255，那麽**如果像素點的多項式值模257後的余數大於255時，我們無法將其直接作為像素值保存**，這部分像素點的數據無法直接傳輸

所以問題的關鍵是如何保存由模數造成的無法傳輸的額外信息

這裏我們選擇一個與256較為接近的模數，即257，這樣我們只需要記錄一種情況，即多項式的值mod257後余256

**我們考慮將這些余256的某通道像素點在圖像數組中的索引放在一個列表中，並將這個列表保存在圖片的元數據（Meta data）中**

### 影子圖像生成

影子圖像生成的步驟如下圖：

![生成秘密圖片](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E7%94%9F%E6%88%90%E7%A7%98%E5%AF%86%E5%9B%BE%E7%89%87.png)

1. 隨機生成多項式系數，並將原像素值（即秘密）作為$a_0$。
2. 將原圖像數組碾平，使得三個通道R、G、B上的值可以同時進行多項式計算，並將計算結果合並為彩色的影子圖像。
3. 如果生成影子圖像時，像素點上的值為256，則將其置0，並保存索引信息到元數據中
4. 遍歷索引，直到生成N個影子圖像

### 原圖像恢復

恢復原圖像時需要先將元數據中存儲的索引數組恢復為一個圖像數組，並與秘密圖像提取的數組相加，這樣就能**恢復余數為256的像素點的信息**

而恢復算法則選用**拉格朗日插值法**

原圖像恢復的步驟如下圖：

![原圖像恢復](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/%E5%8E%9F%E5%9B%BE%E5%83%8F%E6%81%A2%E5%A4%8D.png)

1. 讀取並處理r個影子圖像，每個圖像的三通道值碾平為一維數組
2. 將影子圖像元數據中的額外信息提取並保存，恢復為一個余256像素點組成的一維數組，並**直接加**到影子圖像的數組上
3. 每次從r個影子圖像數組各讀取一個像素值，使用拉格朗日插值法恢復出原圖像的一個像素值

### 創新點

其實一開始我有考慮過將程序改造為多線程，並將其作為一個創新點。但是後來我意識到，**秘密共享其實是一個計算密集型的過程**。對於python的並發編程來說，I/O密集型的程序比計算密集型的程序更能充分利用多線程的好處。**在本任務中使用多線程，對速度的提升不大**。

#### 無損恢復

本算法**最大的創新點**，就是**實現了無損的圖像秘密共享過程，並且生成的影子圖像的大小不會太大**

較為常見的模251和模257都會不同程度上地導致原圖像信息的丟失

如果我們能將模256的信息額外保存在圖片的元數據中，就能實現無損恢復

將中間影子圖像的格式選為png，我們能夠快速的將額外信息存在文件的chunk中

``` python
def insert_text_chunk(src_png, dst_png, text):
    '''在png中的第二個chunk插入自定義內容'''
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()#創建一個每次返回一個chunk的生成器
    chunk_list = list(chunks)
    chunk_item = tuple([b'tEXt', text])
    index = 1
    chunk_list.insert(index, chunk_item)

    with open(dst_png, 'wb') as dst_file:
        png.write_chunks(dst_file, chunk_list)

def read_text_chunk(src_png, index=1):
    '''讀取png的第index個chunk'''
    reader = png.Reader(filename=src_png)
    chunks = reader.chunks()
    chunk_list = list(chunks)
    img_extra = chunk_list[index][1].decode()
    img_extra = eval(img_extra)
    return img_extra
```

那麽選擇存入元數據的額外信息需要保證能有**很高的信息密度**，很顯然將**余256的索引存在一個數組***是個合適的選擇

通過numpy提供的**where()方法**，我們可以快速的從數組中找到值為256的元素的索引，並利用該索引置0

``` python
indices = np.where(secret_img == 256)[0]
img_extra = indices.tolist()
secret_img[indices] = 0
```

在恢復圖像時，我們只需要將影子圖像的數組與從額外信息恢復的同尺寸的數組相加即可：

``` python
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

#### 一個py文件實現所有功能

而另一個創新點，則是將整個秘密共享過程集成在一個python源碼中。在控製臺運行該py文件時，**通過設置選項和傳入不同的參數**，我們能夠完成三種任務：影子圖像生成、原圖像恢復以及圖像像素值對比

這種一站式的解決方案極大地簡化了操作流程，用戶無需切換不同的程序或腳本即可完成整個秘密共享的周期。

程序中的關鍵選項說明如下：

- `-e` / `--encode`：這個選項後跟原始圖像的路徑，用於指定需要進行秘密共享加密的圖像文件。
- `-d` / `--decode`：這個選項後跟解密後的圖像的保存路徑，用於指定解密操作的輸出目錄。
- `-n`：這個選項後跟的參數設置了要生成的影子圖像的總數，即秘密共享的分片數。
- `-r`：這個選項後跟的參數設置了重建原始圖像所需的最少影子圖像數，即秘密共享的閾值。
- `-i` / `--index`：這個選項接受一個或多個整數參數，代表用於解密操作的影子圖像的索引。
- `-c` / `--compare`：這個選項後跟兩個圖像文件的路徑，用於比較這兩個圖像的差異。

同時，你還可以使用`-h`參數調出程序的說明書，**這些功能都得益於python的argparse庫**

值得註意的是，解密時影子圖像需和Shamir.py存儲在同一路徑下，並以secret_{index}的規則命名，且格式為PNG

#### 顯示直觀且詳細

通過精心設計的命令行界面，本程序在執行各種操作時，如加密、解密和比較圖像，都會**給出清晰的進度反饋和詳盡的狀態信息**。例如，在解密過程中，程序不僅會**顯示當前處理的進度條**，還會在完成後輸出解密**所用的總時間**，使用戶能夠明確地了解到任務執行的效率。

具體實現如下：

1. **進度條顯示**：在 `decode` 函數中，通過計算當前處理的像素與總像素數的比例，我們實現了一個動態更新的進度條。這個進度條不僅在視覺上給出了解密過程的即時反饋，還通過百分比精確地表達了當前的完成狀態。

   ```python
   percent_done = (i + 1) * 100 // dim
   if last_percent_reported != percent_done:
       if percent_done % 1 == 0:  # 每增加1%更新一次進度
           last_percent_reported = percent_done
           bar_length = 50
           block = int(bar_length * percent_done / 100)
           text = "\r[{}{}] {:.2f}%".format("█" * block, " " * (bar_length - block), percent_done)
           sys.stdout.write(text)
           sys.stdout.flush()
   ```

2. **文件大小的動態顯示**：使用 `get_file_size` 函數，程序在保存每個影子圖像和恢復的原圖像後，都會輸出文件的大小。這個大小是動態計算並格式化的，根據文件的實際大小自動選擇最合適的單位（如B, KB, MB），使信息展示更為直觀。

   ```python
   size = get_file_size(secret_img_path)
   print(f"{secret_img_path} saved.", size)
   ```

3. **圖像比較的詳細報告**：在比較兩個圖像時，`compare_images` 函數不僅輸出了兩圖像的**平均差異值，還輸出了最大差異、最小差異以及差異的標準差**，為用戶提供了全面的圖像差異分析。該**報告的結果能充分說明本算法在無損秘密共享上的可靠性**。

   ```python
   print("Mean difference:", diff_value)
   print("Max difference:", round(np.max(diff), 4))
   print("Min difference:", round(np.min(diff), 4))
   print("Standard deviation of difference:", round(np.std(diff), 4))
   ```

4. **運行時間的直觀顯示**：在程序的關鍵節點，如加密結束或解密完成後，程序會計算並顯示整個操作所花費的時間。這不僅提供了操作的即時反饋，而且還允許用戶對程序的性能進行評估。通過記錄操作開始和結束的時間戳，程序可以輸出精確到毫秒的運行時間，使得性能測試結果更加準確。

   ``` python
   start_time = time.time()  # 操作開始前記錄時間
   # ... 執行操作 ...
   end_time = time.time()    # 操作結束後記錄時間
   print(f"Operation completed. Time elapsed: {end_time - start_time:.2f} seconds.")
   ```

以上功能的實現，確保了用戶在使用本程序時，能夠獲得詳盡的操作信息，包括操作進度、文件大小以及操作耗時等。這些直觀的顯示信息不僅提高了用戶操作的透明度，也增強了用戶對程序性能的信心。

## 實驗結果

在實驗中，我會對程序的**不同功能的執行效果**進行演示，並通過**消融實驗**測試本程序實現的無損模塊**在各方面上的提升**

我會使用我的頭像作為測試樣本，原圖像avatar.png如下

<img src="https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/test.jpg" alt="test" style="zoom:50%;" />

原圖像的大小為243KB，尺寸為640*640

之所以選擇png格式的原圖是因為PNG是一種無損壓縮的圖像格式，這意味著在重新恢復圖像時，像素數據不會發生變化，這更有利於我們精準測試整個過程是不是無損的秘密共享

### 功能測試

首先進行影子圖像的生成，執行如下指令

``` shell
python Shamir.py -e avatar.png -n 5 -r 3 
```

成功生成5張影子圖像

![image-20231111212119557](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212119557.png)

![image-20231111212211000](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212211000.png)

接下來使用圖像恢復功能，執行如下指令

``` shell
python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5
```

我們選用序號為1、4、5的影子圖像來恢復原圖像

![image-20231111212253826](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212253826.png)

![image-20231111212319416](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212319416.png)

最後我們來對比一下恢復得到的圖像與原圖像之間像素值的差別

執行如下命令

``` shell
python Shamir.py -c avatar_recover.png avatar.png
```

![image-20231111212340635](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111212340635.png)

**恢復圖像與原圖像完全一致**，說明成功實現了無損的圖片秘密共享

如果我們想同時執行所有任務，完成加密解密和對比，可以執行下面這條指令

``` shell
python Shamir.py -e avatar.png -n 5 -r 3 -d avatar_recover.png -i 1 4 5 -c avatar_recover.png avatar.png
```

![image-20231111213341368](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111213341368.png)

### 消融實驗

在消融實驗中，我會測試額外信息這個模塊的影響

將源代碼中有關extra部分的內容去掉後，我們再次執行整個過程

下圖是**去掉額外信息模塊**後的算法運行結果和恢復的圖像

![image-20231111213949707](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111213949707.png)

<img src="https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/avatar_recover-16997100367043.png" alt="avatar_recover" style="zoom:50%;" />

下圖是**擁有額外信息模塊**的算法運行結果和恢復的圖像

![image-20231111213701330](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111213701330.png)

<img src="https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/avatar_recover.png" alt="avatar_recover" style="zoom:50%;" />

首先是無損方面，沒有額外信息模塊的算法因為余256像素點的影響，平均像素值差異為0.9862

而擁有額外信息模塊的算法恢復的圖片與原圖完全一致，平均像素值差異為0

![image-20231111214529421](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111214529421.png)

![image-20231111214822083](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111214822083.png)

直接觀察恢復的圖像我們能夠更直觀的發現原因

![image-20231111214852488](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111214852488.png)

![image-20231111220014444](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111220014444.png)

沒有額外信息模塊的算法恢復的圖像中，有許多因為丟失信息而恢復失敗的像素點

上述差別說明**額外信息模塊使得信息都被保留，實現了無損秘密共享**

除了圖像指令，我們還應關註算法的時間以及影子圖像的大小

額外信息模塊中進行了更多的運算以及存儲，我們需要了解其對於用戶體驗的影響

![image-20231111215428807](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215428807.png)

![image-20231111215414634](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215414634.png)

可以發現，添加了額外信息模塊的算法運行時間變化不大，說明我們**在元數據中添加信息的效率非常高，解碼時余256點的恢復效率也很高**

![image-20231111215701758](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215701758.png)

![image-20231111215723298](https://r1ck-blog.oss-cn-shenzhen.aliyuncs.com/image-20231111215723298.png)

添加在元數據中的信息占影子圖像大小的3%左右，說明我們添加的額外信息的**信息密度非常高，保證了無損秘密共享的同時也不會消耗更多的空間資源**

## 貢獻

歡迎對此項目進行貢獻！如果您有改進建議或遇到任何問題，請隨時打開一個問題或提交一個拉取請求。

## 許可證

該項目在GNU通用公共許可證v3.0下獲得許可

詳情請見[LICENSE](../LICENSE)文件。