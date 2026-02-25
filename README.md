[TOC]
# 拼豆图纸生成工坊 (Perler Bead Pattern Generator)

这是一个基于 Flask 的 Web 应用，致力于将用户上传的图片智能转换为拼豆（Perler Beads）图纸。项目集成了图像降色算法、在线像素编辑器、以及完整的用户账户体系（含邮件验证与 VIP 会员机制）。



## 快速开始

### 1. 初始化数据库

首次运行项目前，必须初始化用户数据库：

```bash
python init_user_db.py

```

*如果后续更新了代码导致数据库报错，可运行 `python fix_db.py` 进行无损修复。*

### 2. 配置邮件服务 (关键)

打开 `app.py`，找到 `邮件配置 (SMTP)` 部分，填入你的邮箱信息。
**注意**：`MAIL_PASSWORD` 必须填写邮箱的 **授权码** (Authorization Code)，而非登录密码。

```python
# app.py 示例配置
app.config['MAIL_SERVER'] = 'smtp.qq.com'      # 例如 QQ 邮箱
app.config['MAIL_PORT'] = 465
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USERNAME'] = '你的邮箱@qq.com'
app.config['MAIL_PASSWORD'] = '你的授权码'      # <--- 注意这里！
app.config['MAIL_DEFAULT_SENDER'] = app.config['MAIL_USERNAME']

```

### 3. 启动服务器

```bash
python app.py

```

### 4. 访问应用

打开浏览器访问：`http://localhost:5000`
默认会跳转至登录页面，请先注册账号。

---

## 核心功能

### 1. 用户中心 (login.html)

该页面是系统的门面，采用响应式卡片设计，集成了用户身份验证与权限管理的四大核心模块：

* **多维度登录**：支持通过 **用户名** 或 **注册邮箱** 进行登录。
* **安全校验**：后端将采用 `werkzeug.security` 对输入的密码进行哈希比对，确保数据安全。
* **会话管理**：登录成功后，系统通过 `Flask-Login` 保持用户会话，作为访问图纸编辑页面的通行证。
* **邮箱强绑定**：注册时必须填写有效邮箱，并输入系统发送的 **6位数字验证码** 进行核验。
* **双重密码确认**：前端强制要求输入两次密码并进行一致性校验，防止因输入失误导致无法登录。
* **防止重名**：后端自动检测用户名及邮箱的唯一性，避免账号冲突。
* **VIP 权限激活**：用户输入在第三方平台购买的 **16位唯一激活码** 即可实时升级，解锁 Pro 算法。
* **密码找回**：支持通过注册邮箱接收验证码，并在输入两次新密码确认后完成重置。

### 2. 图片转图纸 (image_conversion)

* **多格式支持**：支持 JPG, PNG, GIF, BMP 等主流图片格式。
* **智能降色算法**：
* **基础版**：基于 KMeans 聚类，快速提取主题色。
* **Pro 进阶版**：结合 CIELAB 色彩空间与空间邻域算法，自动合并占比 <2% 的杂色，显著提升图纸纯净度。


* **自定义参数**：用户可自由设定目标尺寸（网格宽度）、单格像素大小、以及目标颜色数量。

### 3. 在线编辑器 (draw_page)

* **实时交互**：点击网格即可修改颜色。
* **批量替换**：支持“同色一键替换”，快速调整整体色调。
* **裁剪工具**：自定义坐标裁剪图纸区域。
* **撤销机制 (Undo)**：支持 10 步历史记录回滚，防止误操作。
* **自动保存**：编辑过程中的数据会自动保存至 SQLite 数据库。
* **导出下载**：生成包含**网格线**、**坐标轴**和**色号图例统计**的高清图纸图片。

---

## 项目结构

### 文件树状结构

```text
Image_Pixelation/
│  app.py                   # [核心] Flask 后端入口，路由分发与配置
│  db_manager.py            # [核心] 数据库操作封装 (用户管理、图纸存取、验证码、快照撤销)
│  image_utils.py           # [核心] 图像处理算法 (KMeans, CIELAB转换, 绘图渲染)
│  file_manager.py          # 文件清理工具 (自动清理旧缓存)
│  init_user_db.py          # 数据库初始化脚本 (建表用)
│  fix_db.py                # 数据库修复脚本 (补全缺失表)
│  debug_mail.py            # 邮件发送调试脚本
│  readData.py              # 颜色库迁移工具 (RGB -> Lab)
│  README.md                # 项目说明文档
│
├─data/                     # 数据存储目录
│  ├─Color/                 # 存放色卡数据库 (colors.db)
│  ├─DrawingData/           # 存放生成的图纸源文件 (.db) 及撤销快照
│  └─users.db               # 用户信息与验证码数据库 (由脚本自动生成)
│
├─font/                     # 字体文件 (用于图纸渲染)
│
├─static/
│  ├─outputs/               # 转换结果图片缓存
│  └─uploads/               # 上传原图缓存
│
└─templates/                # 前端页面
       index.html           # 首页 / 仪表盘
       login.html           # 登录 / 注册 / 找回密码 / VIP激活
       image_conversion.html# 图片转换设置页
       draw_page.html       # 在线绘图编辑器
       colors.html          # 色卡管理页

```

### 核心脚本说明

* **app.py**: Flask 后端主程序，负责路由分发和各类 API 接口。
* **db_manager.py**: 负责图纸数据库的创建、裁剪、单格/批量颜色更新，以及最多 10 步的滚动快照撤销 (`undo`) 逻辑。
* **file_manager.py**: 文件管理器，用于限制 `static/uploads` 和 `static/outputs` 文件夹内的文件数量（默认最多保留20个），按照修改时间自动清理旧文件。
* **image_utils.py**: 核心图像处理工具。包含图片转像素网格、KMeans 聚类降色、基于 CIELAB 空间的 Pro 版杂色合并算法，以及图纸的可视化渲染（带网格线和图例）。
* **readData.py**: 数据库迁移脚本。用于读取颜色的 RGB 值，将其转换为 Lab 色彩空间值，并覆盖更新到 `colors.db` 中。

---

## 数据库结构

### 1. 用户数据库 (users.db)

**表名**: `users` (存储用户信息)
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **id** | INTEGER | 主键，自动递增。 |
| **username** | TEXT | 用户名，全站唯一 (UNIQUE)。 |
| **email** | TEXT | 用户邮箱，用于找回密码及身份标识 (UNIQUE)。 |
| **password_hash** | TEXT | 经过哈希加密后的密码存储。 |
| **user_level** | TEXT | 用户等级：`common` (普通) 或 `vip` (VIP)。 |
| **vip_expire_at** | DATETIME | VIP 过期时间 (VIP 用户特有)。 |
| **created_at** | DATETIME | 账户注册时间，默认为 `CURRENT_TIMESTAMP`。 |

**表名**: `vip_codes` (VIP 卡密)
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **code** | TEXT | 16位随机卡密 (PRIMARY KEY)。 |
| **is_used** | INTEGER | 使用状态：0 为未使用，1 为已使用。 |
| **used_by** | TEXT | 使用该卡密的用户用户名。 |
| **valid_days** | INTEGER | 该卡密提供的 VIP 天数（如 31 天或 365 天）。 |

**表名**: `email_verify` (邮箱验证码)
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **email** | TEXT | 目标邮箱。 |
| **code** | TEXT | 6位数字验证码。 |
| **expire_at** | DATETIME | 过期时间（通常为生成后的 5 分钟）。 |

### 2. 颜色库 (colors.db)

用于存储可用的标准颜色色版。
**表名**: `colors`
**字段**: `num` (TEXT/INT), `R`, `G`, `B`, `lab_l`, `lab_a`, `lab_b`

### 3. 图纸源数据库 ({drawing_id}.db)

用于存储单张图纸的像素排布和统计元数据。
**表名**: `grid`
| 字段名 | 类型 | 说明 |
| --- | --- | --- |
| **r** | INTEGER | 行坐标 |
| **c** | INTEGER | 列坐标 |
| **color_id** | TEXT | 颜色编号 |

**表名**: `metadata`

* `key='dimensions'`: value 为 `[rows, cols]` 的 JSON 字符串。
* `key='color_code_count'`: value 为包含颜色 ID 及其详情的 JSON 字典。

---

## 环境依赖

本项目基于 Python 3.8+ 开发。

### 安装依赖库

请在终端运行以下命令安装所需库：

```bash
pip install flask flask-login flask-mail pillow numpy pandas opencv-python scikit-image scikit-learn

```

### 特别注意：scikit-learn 版本兼容性

在 `image_utils.py` 中，KMeans 聚类函数的初始化参数需注意版本兼容性。
**错误写法 (新版 sklearn 不支持)**: `n_init='auto'`
**正确写法**: `n_init=10`

```python
kmeans = KMeans(
    n_clusters=int(target_cluster_count), 
    random_state=0, 
    n_init=10  # <--- 请确保此处为整数
).fit(lab_data)

```

---

## 配置说明

### Windows 主机名报错修复

在 Windows 环境下，`socket` 模块可能因中文计算机名导致邮件发送失败。本项目已在 `app.py` 头部集成自动修复补丁：

```python
import socket
try:
    socket.gethostname = lambda: "localhost"
except:
    pass

```

### 自动清理机制

为了防止磁盘占满，`file_manager.py` 会在每次生成新图纸时，自动检查 `static/uploads` 和 `static/outputs` 文件夹，仅保留最新的 20 个文件。

---

## 常见问题排查

**Q1: 点击发送验证码没反应？**

* 检查终端是否有报错信息。
* 如果是 `ConnectionRefused` 或 `Timeout`，请检查网络是否拦截了 465 端口。
* 如果是数据库错误，请确保运行了 `init_user_db.py`。

**Q2: 邮件发送报错 `535 Error: authentication failed`？**

* 这是授权码错误。请去邮箱设置里重新生成一个新的授权码，替换到 `app.py` 中。

**Q3: 注册时提示 `Database is locked`？**

* SQLite 是单文件数据库，不支持高并发写入。请确保没有使用 DB Browser 等软件同时打开 `users.db` 并在编辑模式下。

**Q4: 图纸生成全是黑色或颜色不对？**

* 请确保 `data/Color/colors.db` 存在且数据完整。如果缺失，请从备份恢复或运行 `readData.py` 重建颜色库。