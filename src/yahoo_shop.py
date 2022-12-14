import time

from selenium import webdriver
from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.keys import Keys as keys
from selenium.webdriver.support.ui import WebDriverWait

MAX_PAGE = 5

query = '絵画'
url = "https://shopping.yahoo.co.jp/search?p={}".format(query)
driver = fs.Service(executable_path="chromedriver.exe")
browser = webdriver.Chrome(service=driver)
browser.get(url)

element = WebDriverWait(browser,30).until(lambda x: x.execute_script("return document.getElementsByClassName('Sq9SEWjoMWXj')[0]"))
element.click()

win_height = browser.execute_script("return window.innerHeight")
print(win_height)

#スクロール開始位置の初期値（ページの先頭からスクロールを開始する）
last_top = 1

page = 0
#ページの最下部までスクロールする無限ループ
while True and page < MAX_PAGE:

  #スクロール前のページの高さを取得
  last_height = browser.execute_script("return document.body.scrollHeight")
  
  #スクロール開始位置を設定
  top = last_top

  #ページ最下部まで、徐々にスクロールしていく
  while top < last_height:
    top += int(win_height * 0.8)
    browser.execute_script("window.scrollTo(0, %d)" % top)
    time.sleep(0.5)

  #１秒待って、スクロール後のページの高さを取得する
  time.sleep(1)
  new_last_height = browser.execute_script("return document.body.scrollHeight")

  #スクロール前後でページの高さに変化がなくなったら無限スクロール終了とみなしてループを抜ける
  if last_height == new_last_height:
    break

  #次のループのスクロール開始位置を設定
  last_top = last_height

  page += 1