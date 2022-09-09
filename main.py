# This is a sample Python script.
from plyfile import PlyData
from glob import glob
from tqdm import tqdm
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def get_num_points(path):
    # Use a breakpoint in the code line below to debug your script.
    pcd = PlyData.read(path)
    num = pcd['vertex'].count
    return num

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    min = 10000000000000000
    files = glob("C:/Users/Diana/Desktop/DATA/Kitti360/data_3d_semantics/train/"+ "\\*\\static\\*.ply")
    for f in tqdm(files):
        temp = get_num_points(f)
        if temp < min:
            min = temp
    print(min)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
