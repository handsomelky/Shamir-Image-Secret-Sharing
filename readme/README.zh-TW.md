# 圖像秘密共享 - Shamir's Secret Sharing

本專案實現了基於 Shamir Secret Sharing 的無損圖像秘密共享。程式可以將一張圖像拆分為多張 PNG 份額圖像，並使用任意滿足閾值數量的份額恢復原始圖像。

[English](../README.md) | [简体中文](README.zh-CN.md) | 繁體中文

## 功能特性

- 逐位元組無損恢復原始圖像檔案。
- 在 GF(256) 上直接對圖像 payload 位元組執行 Shamir 共享。
- 使用緊湊的灰度 PNG 作為份額容器，而不是生成同尺寸雜訊圖。
- 當 zlib 壓縮能減小 payload 時自動使用壓縮。
- 使用密碼學安全隨機數生成多項式係數。
- 使用 NumPy 向量化恢復，提升解碼性能。
- 使用 Rich 提供狀態面板、進度條、結果表格和錯誤提示。
- 支援配置份額輸出目錄、份額讀取目錄和份額檔名前綴。
- 內建圖像比較命令和單元測試。

## 安裝

默認使用 Python 3.12，並使用 `uv` 管理環境和依賴。依賴聲明在 `pyproject.toml` 中。

```shell
uv sync
```

運行時依賴：

- NumPy
- Pillow
- Rich

## 使用方法

通過 uv 運行 CLI：

```shell
uv run python Shamir.py [options]
```

命令行參數：

| 參數 | 用法 |
| --- | --- |
| `-e`, `--encode <圖片路徑>` | 將輸入圖像編碼為份額圖像。需要同時傳入 `-n` 和 `-r`。 |
| `-d`, `--decode <輸出路徑>` | 恢復原始圖像並保存到指定路徑。需要同時傳入 `-r` 和 `-i`。 |
| `-n <份額數量>` | 要生成的份額總數。必須大於等於 `r`，且不能超過 255。 |
| `-r <恢復閾值>` | 恢復原圖所需的最少份額數量。必須至少為 2。 |
| `-i`, `--index <份額索引...>` | 解碼時使用的份額索引，例如 `-i 1 4 5`。索引必須互不重複，並位於 `1..255`。 |
| `-c`, `--compare <圖像A> <圖像B>` | 比較兩張圖像，輸出像素差異的均值、最大值、最小值和標準差。 |
| `--output-dir <目錄>` | 編碼時份額圖像的輸出目錄。默認是當前目錄。 |
| `--share-dir <目錄>` | 解碼時讀取份額圖像的目錄。默認是當前目錄。 |
| `--share-prefix <前綴>` | 份額檔名前綴。默認是 `secret`，生成如 `secret_1.png` 的檔名。 |
| `-h`, `--help` | 顯示完整命令行幫助。 |

## 示例

```shell
# 生成 5 個份額，恢復閾值為 3
uv run python Shamir.py -e avatar.png -n 5 -r 3 --output-dir shares

# 使用份額 1、4、5 恢復圖像
uv run python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5 --share-dir shares

# 檢查是否無損恢復
uv run python Shamir.py -c avatar.png avatar_recover.png
```

## 算法說明

程式先使用 Pillow 校驗輸入確實是圖像，然後將原始圖像檔案位元組作為 secret payload。只有當 zlib 壓縮能讓 payload 變小時，程式才會保存壓縮後的 payload。

算法在有限域 GF(256) 上運行，因此每個份額值天然都是 0 到 255 的位元組。對於每個 payload 位元組 `s`，編碼器構造隨機多項式：

```text
f(x) = s + a1*x + a2*x^2 + ... + a(r-1)*x^(r-1) in GF(256)
```

對於份額索引 `x`，保存的份額位元組為 `f(x)`。這些隨機份額位元組會被打包成灰度 PNG 圖像。份額 PNG 的寬高只是 payload 容器佈局，不再對應原圖寬高。

## 份額容器格式

每個份額 PNG 都是灰度位元組容器。重建出的 payload 以內部二進制頭開頭，包含魔數、壓縮標誌、原始位元組長度和儲存位元組長度。為了填滿 PNG 矩形而增加的 padding 位元組會在恢復後被忽略。

## 圖像恢復

恢復時，解碼器會：

1. 讀取選中的灰度份額 PNG，並將其位元組碾平。
2. 根據選中的份額索引一次性計算拉格朗日權重。
3. 使用 NumPy 向量化 GF(256) 運算恢復 payload 位元組。
4. 解析 payload 頭，按需解壓，並寫回原始圖像檔案位元組。

恢復至少需要 `r` 個互不重複的份額。份額索引必須在 1 到 255 之間，並且所有選中的份額圖像必須擁有相同的編碼位元組長度。

## 參數校驗

CLI 會校驗：

- `r >= 2`
- `n >= r`
- `n <= 255`
- 解碼索引存在、互不重複，並且位於 `1..255`
- 選中的份額是灰度 PNG，且編碼位元組長度一致
- 重建出的 payload 頭和長度有效
- 被比較的兩張圖像形狀一致

## 測試

運行測試：

```shell
uv run python -m unittest discover
```

測試覆蓋 RGB 圖像往返恢復、GF(256) 重建、payload 解析、參數校驗、非法解碼索引、份額容器行為和圖像比較尺寸檢查。

## 專案結構

```text
Shamir.py              主 CLI 和算法實現
pyproject.toml         專案元數據和依賴
tests/                單元測試
readme/               多語言 README
```

## 貢獻

歡迎貢獻程式碼、測試、文檔或實現改進。可以提交 issue 或 pull request。

## 許可證

本專案基於 GNU General Public License v3.0 許可發布。詳情見 [LICENSE](../LICENSE)。
