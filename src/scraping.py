import requests
from time import sleep

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import os
import re
import pickle
import sys 
sys.path.append("lib.bs4")

LIMIT_NUM = 100
SEARCH_PAGE_NUM = 5

def download_image(url, file_path):
  r = requests.get(url, stream=True)
  if r.status_code == 200:
    with open(file_path, "wb") as f:
      f.write(r.content)
    
import base64
def save_base64_image(data, file_path):
  # base64の読み込みは4文字ごとに行う。4文字で区切れない部分は「=」で補う
  data = data + '=' * (-len(data) % 4)
  img = base64.b64decode(data.encode())
  with open(file_path, "wb") as f:
      f.write(img)

def search_page(driver,url):
  img_urls = []
  label_price = []
  drop_idx = []

    # すべての要素が読み込まれるまで待つ。タイムアウトは15秒。
  WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)

  driver.get(url)
  html = driver.page_source.encode("utf-8")
  soup = BeautifulSoup(html, "html.parser")

  #値段取得
  span_tags = soup.find_all("span", class_='a8Pemb OFFNJ', limit=LIMIT_NUM)
  for span in span_tags:
    s = ''.join(filter(str.isalnum, span.text))
    label_price.append(int(s))

  #画像取得
  div_tags = soup.find_all("div", attrs={'class':'ArOc1c'}, limit=LIMIT_NUM)
  for idx, div_tag in enumerate(div_tags):
    for img_tag in div_tag.children:
      url = img_tag.get("src")

    if url is None:
      url = img_tag.get("data-src")

    if url is not None:
      img_urls.append(url)
    else:
      drop_idx.append(idx)
  
  for tmp in reversed(drop_idx):
    label_price.pop(tmp)

  np_tag = soup.find("a", attrs={'id':'pnnext'})
  try:
    np_url = np_tag.get("href")
  except:
    np_url = False

  return img_urls, label_price, np_url

def data_load(query):
  options = webdriver.ChromeOptions()
  options.add_argument('--headless')
  options.add_argument('--no-sandbox')
  options.add_argument('--disable-dev-shm-usage')
  driver = webdriver.Chrome('chromedriver',options=options)
  driver.implicitly_wait(10)

  img_urls = []
  label_price = []
  flag = True
  url = "https://www.google.com/search?q={}&hl=ja&tbm=shop&num=100".format(query)

  page_num = 0
  while(flag):
    page_num += 1 
    #ページ内で検索（npタグがあったら繰り返し）
    one_img_url, one_label_price, np_url = search_page(driver=driver,url=url)

    img_urls.extend(one_img_url)
    label_price.extend(one_label_price)

    if np_url == False or page_num >= SEARCH_PAGE_NUM:
      break
    else:
      url = "https://www.google.com/{}".format(np_url)

  save_dir = "../data/fig/"
  if not os.path.exists(save_dir):
      os.mkdir(save_dir)

  base64_string = "data:image/jpeg;base64,"
  png_base64_string = "data:image/png;base64,"

  for index, url in enumerate(img_urls):
    file_name = "{}.jpg".format(index)

    image_path = os.path.join(save_dir, file_name)

    if len(re.findall(base64_string, url)) > 0 or len(re.findall(png_base64_string, url)) > 0:
      url = url.replace(base64_string, "")
      save_base64_image(data=url, file_path=image_path)
    else:
      download_image(url=url, file_path=image_path)

  print('label_price len:',len(label_price))
  with open('../data/label/price.pickle', 'wb') as f:
    pickle.dump(label_price, f)

  driver.quit()


import cv2
def resize_fig(fig_num):
    resize_h = 400
    resize_w = 400

    for i in range(fig_num):
        input_path = '../data/fig/{}.jpg'.format(i)
        output_path = '../data/reshape_fig/{}.jpg'.format(i)
        im = cv2.imread(input_path)
        im = cv2.resize(im, dsize = (resize_h, resize_w))
        cv2.imwrite(output_path, im)
    
    print("size to reshape = (", resize_h, "," ,resize_w, ")")


if __name__ == '__main__':
  data_load('絵画')