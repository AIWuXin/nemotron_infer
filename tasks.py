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


@task
def gen_stub(c):
    """从 binding.pyd 生成 .pyi 类型存根（IDE 补全 / 类型检查用）。

    pybind11-stubgen 0.x 把每个模块当包，产出 <module>/__init__.pyi；这里抽出来拍平成
    nemotron_infer/csrc/binding.pyi（与 .pyd 同目录的 PEP 561 内联存根，类型检查器会优先采用）。
    前置：先编译出 binding.*.pyd（cmake --build ... --target nemotron_infer）。
    """
    import shutil
    from pathlib import Path

    module = "nemotron_infer.csrc.binding"
    tmp = Path("_stubgen_tmp")
    dst = Path("nemotron_infer/csrc/binding.pyi")

    if tmp.exists():
        shutil.rmtree(tmp)
    # --root-module-suffix "" 让临时目录镜像真实包路径，便于定位产物
    c.run(
        'uv run --no-sync python -m pybind11_stubgen '
        '--ignore-invalid=all --no-setup-py --root-module-suffix "" '
        f'-o {tmp} {module}'
    )

    src = tmp / "nemotron_infer" / "csrc" / "binding" / "__init__.pyi"
    if not src.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"未找到生成的存根: {src}（binding.pyd 是否已编译并可导入？）")

    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"已生成类型存根: {dst}")
