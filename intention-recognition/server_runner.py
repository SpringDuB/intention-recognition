import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from utils.server import app

# 0.0.0.0 不是实际有效的IP地址，表示当前机器的所有ip地址
app.run("0.0.0.0", port=9999)
