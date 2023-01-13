import pickle
import re
import sys

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

sys.path.append("lib.bs4")

SEARCH_PAGE_MAX = 1000 # 1ページあたり約50画像
COLLECT_FIG_NUM = 10000
save_data = []

def search_page(driver,url):
  # すべての要素が読み込まれるまで待つ。タイムアウトは15秒。
  WebDriverWait(driver, 15).until(EC.presence_of_all_elements_located)

  driver.get(url)
  html = driver.page_source.encode("utf-8")
  soup = BeautifulSoup(html, "html.parser")

  item_preview_tags = soup.find_all("div", class_="item-preview")

  for item_preview_tag in item_preview_tags:
    img_src = item_preview_tag.find("img").get("src")
    amount = item_preview_tag.find("span", class_="amount")
    #eg.) amount = <span class="amount">￥20,000 <small> (税込み) </small></span>

    #amountの数字だけ取り出す
    price = re.sub(r'\D', '', amount.text)
    #amount = '展示中', 'sold'で価格がついていない場合は考慮しない
    if len(price) == 0:
      continue

    # 価格が100,000円を超える場合は外れ値として除外
    if int(price) >= 100000:
      continue

    save_data.append({'src': img_src, 'price': int(price)})

def data_load():
  options = webdriver.ChromeOptions()
  options.add_argument('--headless')
  options.add_argument('--no-sandbox')
  options.add_argument('--disable-dev-shm-usage')
  options.add_argument('--disable-gpu')
  driver = webdriver.Chrome('chromedriver',options=options)
  driver.implicitly_wait(10)

  
  for page_num in range(SEARCH_PAGE_MAX):
    url = "https://thisisgallery.com/products/page/{}".format(page_num)
    try:
      search_page(driver=driver,url=url)
    except:
      break
    
    if len(save_data) >= COLLECT_FIG_NUM:
      break
  
  driver.quit()

  df = pd.DataFrame(save_data)
  with open('../data/this_is_gallery/df.pickle', 'wb') as f:
    pickle.dump(df, f)
  
  print(df)
  print(len(df))

if __name__ == '__main__':
  data_load()