import os

from skrobot.data import fetch_urdfpath, pr2_urdfpath

# bit dirty, but we will probably use only pr2 and fetch, so...

pr2_urdf_path = os.path.expanduser("~/.skrobot/pr2_description")
if not os.path.exists(pr2_urdf_path):
    print("downloading pr2 model... This takes place only once.")
    pr2_urdfpath()

pr2_urdf_path = os.path.expanduser("~/.skrobot/fetch_description")
if not os.path.exists(pr2_urdf_path):
    print("downloading fetch model... This takes place only once.")
    fetch_urdfpath()
