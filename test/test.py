import sys
import pkg_resources

def test_example():
    # 简单的断言测试
    assert 1 + 1 == 2

def test_python_version():
    # 测试 Python 版本是否可用
    python_version = sys.version
    python_executable = sys.executable
    assert python_version is not None
    assert python_executable is not None
    print(f"Python version: {python_version}")
    print(f"Python executable: {python_executable}")

def test_installed_packages():
    # 测试是否能够获取已安装包的信息
    packages = list(pkg_resources.working_set)
    assert len(packages) > 0  # 确保至少有一个包被安装
    print("\nInstalled packages and versions:")
    for dist in packages:
        print(f"{dist.project_name}=={dist.version}")

def test_write_packages_txt():
    # 测试写入安装包信息到文件
    file_path = "packages.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        # 写入 Python 版本信息
        f.write(f"Python version: {sys.version}\n")
        f.write(f"Python executable: {sys.executable}\n")
        # 写入已安装的包信息
        f.write("Installed packages and versions:\n")
        for dist in pkg_resources.working_set:
            f.write(f"{dist.project_name}=={dist.version}\n")

    # 检查文件是否成功创建并包含内容
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Python version" in content
    assert "Installed packages and versions" in content
