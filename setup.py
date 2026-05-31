from setuptools import Command, find_packages, setup

__lib_name__ = "SpaWeaver"
__lib_version__ = "0.0.3"
__description__ = "test"
__url__ = ""
__author__ = "Yonghao Liu and Chuyao Wang"
__author_email__ = "yonghao20@mails.jlu.edu.cn"
__license__ = "MIT"
__requires__ = ["requests",]

with open("README.md", "r", encoding="utf-8") as f:
    __long_description__ = f.read()

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["SpaWeaver"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True,
    long_description = '''test''',
    long_description_content_type="text/markdown"
)

