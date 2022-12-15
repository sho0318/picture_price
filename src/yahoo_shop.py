import pickle
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys as keys
from selenium.webdriver.support.ui import WebDriverWait

MAX_PAGE = 100

query = '絵画+インテリア'
url = "https://shopping.yahoo.co.jp/search?p={}&view=fg".format(query)

page = 0
cnt_fig = 0 
flag = True
fig_array = []
while flag and page < MAX_PAGE:

  driver = fs.Service(executable_path="chromedriver.exe")
  browser = webdriver.Chrome(service=driver)
  browser.get(url)

  try:
    element = WebDriverWait(browser,30).until(lambda x: x.execute_script("return document.getElementsByClassName('Sq9SEWjoMWXj')[0]"))
    element.click()
  except:
    pass

  win_height = browser.execute_script("return window.innerHeight")

  last_top = 1

#ページの最下部までスクロールする無限ループ
  while page < MAX_PAGE:

    last_height = browser.execute_script("return document.body.scrollHeight")
    
    #スクロール開始位置を設定
    top = last_top

    #ページ最下部まで、徐々にスクロールしていく
    while top < last_height:
      top += int(win_height * 0.8)
      browser.execute_script("window.scrollTo(0, %d)" % top)
      time.sleep(0.4)

    time.sleep(1.5)
    new_last_height = browser.execute_script("return document.body.scrollHeight")

    if last_height == new_last_height:
      try:
        button = browser.find_elements(By.CLASS_NAME, "Button.Button--gray.Button--clickable.aKukkX--BGi9")[-1]
        if button.find_element(By.CLASS_NAME, 'zjvEY0rZSt-X').text == '次の30件':
          url = button.get_attribute("href")
        else:
          print('search to the last page')
          flag = False
        break
      except:
        print('there is no button')
        flag = False
        break

    last_top = last_height

    page += 1

  fig_wrappers = browser.find_elements(By.CLASS_NAME, "HXAkQnwvR-zR")
  for i,fig_wrapper in enumerate(fig_wrappers):
    fig_src = fig_wrapper.find_elements(By.CLASS_NAME, "_9rn9ieDRENke")[-1].get_attribute("src")
    if fig_src == 'https://s.yimg.jp/i/space.gif':
      # fig_src = fig_wrapper.find_element(By.CLASS_NAME, "LazyImage__main").get_attribute("src")
      continue

    label = fig_wrapper.find_element(By.CLASS_NAME, "L6licUwI07IZ").text
    label = label.replace(",","")
    fig_array.append([fig_src, label])
  
  cnt_fig += i+1
  
  print("----------------",len(fig_array),"/",cnt_fig," page:",page)

df = pd.DataFrame(fig_array, columns=['src','label'])
print(df)
print(df.info())

with open('../data/yahoo_shop/df.pickle', "wb") as f:
  pickle.dump(df, f)