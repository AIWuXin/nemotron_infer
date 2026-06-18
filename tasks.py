from invoke import task
import sys
import tomllib  # Python 3.11+ 内置，旧版用 import tomli

@task
def retag(c, version=None):
    """修改 wheel 的平台标签"""

    # 如果没传版本号，从 pyproject.toml 读取
    if version is None:
        with open("pyproject.toml", "rb") as f:
            pyproject = tomllib.load(f)
        version = pyproject["project"]["version"]

    if sys.platform == "win32":
        platform_tag = "win_amd64"
    elif sys.platform == "linux":
        platform_tag = "manylinux_2_17_x86_64"
    elif sys.platform == "darwin":
        platform_tag = "macosx_10_15_x86_64"
    else:
        raise RuntimeError(f"不支持的平台: {sys.platform}")

    c.run(
        f"uv run python -m wheel tags "
        f"--python-tag cp312 "
        f"--abi-tag cp312 "
        f"--platform-tag {platform_tag} "
        f"./dist/nemotron_infer-{version}-py3-none-any.whl"
    )

@task
def build(c):
    """构建并在成功后执行自定义操作"""
    c.run("uv build")
    retag(c)  # 自动从 pyproject.toml 读取版本
    c.run("echo '构建完成'")
