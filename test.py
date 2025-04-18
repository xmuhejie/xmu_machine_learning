import sys
import pkg_resources

def test_py():
    assert 1 + 1 == 2
# 输出Python版本
print(f"Python version: {sys.version}")
# 输出Python解释器路径
print(f"\nPython executable: {sys.executable}")
# 输出已安装的包及其版本
print(f"\nInstalled packages and versions:")
for dist in pkg_resources.working_set:
    print(f"{dist.project_name}=={dist.version}")

# 写入txt文件保存安装的包及其版本
with open("packages.txt", "w", encoding="utf-8") as f:
    # 写入python版本信息
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Python executable: {sys.executable}\n")
    # 写入已安装的包信息
    f.write(f"Installed packages and versions:\n")
    for dist in pkg_resources.working_set:
        f.write(f"{dist.project_name}=={dist.version}\n")
