# 图像秘密共享 - Shamir's Secret Sharing

本项目实现了基于 Shamir Secret Sharing 的无损图像秘密共享。程序可以将一张图像拆分为多张 PNG 份额图像，并使用任意满足阈值数量的份额恢复原始图像。

[English](../README.md) | [繁體中文](README.zh-TW.md) | 简体中文

## 功能特性

- 逐字节无损恢复原始图像文件。
- 在 GF(256) 上直接对图像 payload 字节执行 Shamir 共享。
- 使用紧凑的灰度 PNG 作为份额容器，而不是生成同尺寸噪声图。
- 当 zlib 压缩能减小 payload 时自动使用压缩。
- 使用密码学安全随机数生成多项式系数。
- 使用 NumPy 向量化恢复，提升解码性能。
- 使用 Rich 提供状态面板、进度条、结果表格和错误提示。
- 支持配置份额输出目录、份额读取目录和份额文件名前缀。
- 内置图像比较命令和单元测试。

## 安装

默认使用 Python 3.12，并使用 `uv` 管理环境和依赖。依赖声明在 `pyproject.toml` 中。

```shell
uv sync
```

运行时依赖：

- NumPy
- Pillow
- Rich

## 使用方法

通过 uv 运行 CLI：

```shell
uv run python Shamir.py [options]
```

命令行参数：

| 参数 | 用法 |
| --- | --- |
| `-e`, `--encode <图片路径>` | 将输入图像编码为份额图像。需要同时传入 `-n` 和 `-r`。 |
| `-d`, `--decode <输出路径>` | 恢复原始图像并保存到指定路径。需要同时传入 `-r` 和 `-i`。 |
| `-n <份额数量>` | 要生成的份额总数。必须大于等于 `r`，且不能超过 255。 |
| `-r <恢复阈值>` | 恢复原图所需的最少份额数量。必须至少为 2。 |
| `-i`, `--index <份额索引...>` | 解码时使用的份额索引，例如 `-i 1 4 5`。索引必须互不重复，并位于 `1..255`。 |
| `-c`, `--compare <图像A> <图像B>` | 比较两张图像，输出像素差异的均值、最大值、最小值和标准差。 |
| `--output-dir <目录>` | 编码时份额图像的输出目录。默认是当前目录。 |
| `--share-dir <目录>` | 解码时读取份额图像的目录。默认是当前目录。 |
| `--share-prefix <前缀>` | 份额文件名前缀。默认是 `secret`，生成如 `secret_1.png` 的文件名。 |
| `-h`, `--help` | 显示完整命令行帮助。 |

## 示例

```shell
# 生成 5 个份额，恢复阈值为 3
uv run python Shamir.py -e avatar.png -n 5 -r 3 --output-dir shares

# 使用份额 1、4、5 恢复图像
uv run python Shamir.py -d avatar_recover.png -r 3 -i 1 4 5 --share-dir shares

# 检查是否无损恢复
uv run python Shamir.py -c avatar.png avatar_recover.png
```

## 算法说明

程序先使用 Pillow 校验输入确实是图像，然后将原始图像文件字节作为 secret payload。只有当 zlib 压缩能让 payload 变小时，程序才会保存压缩后的 payload。

算法在有限域 GF(256) 上运行，因此每个份额值天然都是 0 到 255 的字节。对于每个 payload 字节 `s`，编码器构造随机多项式：

```text
f(x) = s + a1*x + a2*x^2 + ... + a(r-1)*x^(r-1) in GF(256)
```

对于份额索引 `x`，保存的份额字节为 `f(x)`。这些随机份额字节会被打包成灰度 PNG 图像。份额 PNG 的宽高只是 payload 容器布局，不再对应原图宽高。

## 份额容器格式

每个份额 PNG 都是灰度字节容器。重建出的 payload 以内部二进制头开头，包含魔数、压缩标志、原始字节长度和存储字节长度。为了填满 PNG 矩形而增加的 padding 字节会在恢复后被忽略。

## 图像恢复

恢复时，解码器会：

1. 读取选中的灰度份额 PNG，并将其字节碾平。
2. 根据选中的份额索引一次性计算拉格朗日权重。
3. 使用 NumPy 向量化 GF(256) 运算恢复 payload 字节。
4. 解析 payload 头，按需解压，并写回原始图像文件字节。

恢复至少需要 `r` 个互不重复的份额。份额索引必须在 1 到 255 之间，并且所有选中的份额图像必须拥有相同的编码字节长度。

## 参数校验

CLI 会校验：

- `r >= 2`
- `n >= r`
- `n <= 255`
- 解码索引存在、互不重复，并且位于 `1..255`
- 选中的份额是灰度 PNG，且编码字节长度一致
- 重建出的 payload 头和长度有效
- 被比较的两张图像形状一致

## 测试

运行测试：

```shell
uv run python -m unittest discover
```

测试覆盖 RGB 图像往返恢复、GF(256) 重建、payload 解析、参数校验、非法解码索引、份额容器行为和图像比较尺寸检查。

## 项目结构

```text
Shamir.py              主 CLI 和算法实现
pyproject.toml         项目元数据和依赖
tests/                单元测试
readme/               多语言 README
```

## 贡献

欢迎贡献代码、测试、文档或实现改进。可以提交 issue 或 pull request。

## 许可证

本项目基于 GNU General Public License v3.0 许可发布。详情见 [LICENSE](../LICENSE)。
