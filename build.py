import sys
import os
from cx_Freeze import setup, Executable

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 依赖项
build_exe_options = {
    "packages": [
        "os",
        "sys",
        "PyQt6",
        "PyQt6.QtWidgets",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWebEngineWidgets",
        "PyQt6.QtWebEngineCore",
        "pyncm",
        "pyncm.apis",
        "pyncm.utils",
        "queue",
        "concurrent.futures",
        "threading",
        "time",
        "logging",
        "argparse",
        "re",
        "dataclasses",
        "http",
        "http.client",
        "http.server",
        "json",
        "pathlib",
        "importlib",
        "random",
        "mutagen",
        "tqdm",
        "coloredlogs",
    ],
    "excludes": [
        "tkinter",
        "unittest",
        "email",
        "html",
        "xml",
        "pydoc",
        "doctest",
        "pdb",
        "pytest",
        "setuptools",
        "pip",
    ],
    "include_files": [
        # 图标文件
        ("icon.ico", "icon.ico"),
        # 如果有其他需要包含的文件，在这里添加
        # 例如：("path/to/file", "destination/path")
    ],
    "include_msvcr": True,  # 包含 MSVC 运行时
    "includes": [
        "http",
        "http.client",
        "http.server",
        "PyQt6.QtWebEngineCore",
        "PyQt6.QtWebEngineWidgets",
    ],
    "zip_include_packages": ["*"],  # 将所有包打包到zip文件中
    "zip_exclude_packages": [],  # 不要排除任何包
}

# 目标文件
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # 使用Windows GUI

# 主程序文件路径
main_script = os.path.join(current_dir, "pyncm", "__main__.py")

# 图标文件路径（如果存在）
icon_path = os.path.join(current_dir, "icon.ico")
icon = icon_path if os.path.exists(icon_path) else None

if __name__ == "__main__":
    # 添加build命令到sys.argv如果没有提供
    if len(sys.argv) == 1:
        sys.argv.append("build")
    
    setup(
        name="网易云音乐下载器",
        version="1.0",
        description="网易云音乐下载工具",
        author="PyNCM",
        options={"build_exe": build_exe_options},
        executables=[
            Executable(
                main_script,  # 主程序文件
                base=base,
                target_name="网易云音乐下载器.exe",  # 生成的exe文件名
                icon=icon,  # 图标文件（如果存在）
                copyright="Copyright © 2024",
                shortcut_name="网易云音乐下载器",  # 创建快捷方式的名称
                shortcut_dir="DesktopFolder",  # 在桌面创建快捷方式
            )
        ]
    )
