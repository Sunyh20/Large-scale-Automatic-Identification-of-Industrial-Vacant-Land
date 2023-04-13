import os

#根据LB文件夹TIF删除YX文件夹中的TIF
def SC_TIF(inp1,inp2):
    inp1_List = [i for i in [files for root, dirs, files in os.walk(inp1)][0]]
    inp2_List = [i for i in [files for root, dirs, files in os.walk(inp2)][0]]
    for tif_file1 in inp2_List:
        if tif_file1 not in inp1_List:
            print(inp2+"\\"+tif_file1)
            os.remove(inp2+"\\"+tif_file1)
    return 0

inp1 = r'*\YB\LB'
inp2 = r'*\YB\YX'
SC_TIF(inp1,inp2)

