from scraping import data_load, resize_fig
import os

QUERY = "絵画"

if __name__ == "__main__":
    data_load(QUERY)
    
    DIR = "../data/fig"
    fig_num = len(os.listdir(DIR))
    print("fig_num :", fig_num)
    print("-------------finish scraping------------")
    
    resize_fig(fig_num)
    print("-------------finish resizing------------")
