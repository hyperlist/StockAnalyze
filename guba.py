import requests
from bs4 import BeautifulSoup
import csv, datetime, os
import json
import pandas as pd
import re
import jieba 
headers = {'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Encoding':'gzip,deflate',
    'Accept-Language':'zh-CN,zh;q=0.9',
    'Cache-Control':'max-age=0',
    'Connection':'keep-alive',
    'Cookie':'st_pvi=87732908203428;st_si=12536249509085;qgqp_b_id=9777e9c5e51986508024bda7f12e6544;_adsame_fullscreen_16884=1',
    'Host':'guba.eastmoney.com',
    #'Referer':'http://guba.eastmoney.com/list,600596,f_1.html',
    'Upgrade-Insecure-Requests':'1',
    'User-Agent':'Mozilla/5.0(WindowsNT6.1;Win64;x64)AppleWebKit/537.36(KHTML,likeGecko)Chrome/65.0.3325.181Safari/537.36'}
    
#获取列表页第n页的具体目标信息，由BeautifulSoup解析完成
def get_urls(id, filepath):
    url = 'http://guba.eastmoney.com/list,' + id + '.html'
    q = requests.get(url,headers=headers)
    soup = BeautifulSoup(q.text,'html.parser')
    urllist = soup.findAll('div',{'class':'articleh'})
    savejson = open(filepath +'.json','a',encoding='utf-8')
    
    savecsv = open(filepath +'.csv','a',encoding='utf-8')
    if os.path.getsize(filepath +'.csv')==0:
        savecsv.write('id,created_time,title' + '\n')
    for row in urllist:
        #print(row('span'),len(row('span')))
        if row('span') != None and len(row('span'))==5:
            try:
                span = row('span')
                news = {}
                link = span[2].a.attrs['href'].replace('/news,','').replace('.html','')
                if id == link.split(',')[0]:
                    news['pid'] = link.split(',')[1]
                    news['titel'] = span[2].a.attrs['title']
                    news['author_url'] = span[3].a.attrs['href']
                    news['author_name'] = span[3].font.get_text()
                    news['ptime'] = datetime.datetime.now().strftime('%Y-') + span[4].get_text()
                    line = json.dumps(dict(news), ensure_ascii=False) + "\n"
                    savejson.write(line)
                    sent = ' '.join(list(jieba.cut(news['titel'].replace(',','，'))))
                    savecsv.write(news['pid'] + ',' + news['ptime'] + ',' + sent + '\n')
            except Exception as err:
                print(err)
                pass
    savejson.close
    savecsv.close
    
def clean(filepath):
    df = pd.read_csv(filepath,error_bad_lines=False)
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    datadir = "./data/"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    stocks = csv.DictReader(open(os.path.join(datadir, "stocks.csv"), 'r', encoding='utf-8'))
    for item in list(stocks):
        datapath = os.path.join(datadir, 'ID-'+item['id'])
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        savefile =os.path.join(datapath, datetime.datetime.now().strftime('%Y-%m-%d'))
        get_urls(item['id'], savefile)
        clean(savefile+'.csv')

