import requests
from bs4 import BeautifulSoup
import csv, datetime, os
import json
import re

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
def get_urls(url, filename):
    q = requests.get(url,headers=headers)
    soup = BeautifulSoup(q.text,'html.parser')
    urllist = soup.findAll('div',{'class':'articleh'})
    savejson = open(filename +'.json','w',encoding='utf-8')
    savecsv = open(filename +'.csv','w',encoding='utf-8')
    for row in urllist:
        #print(row('span'),len(row('span')))
        if row('span') != None and len(row('span'))==5:
            try:
                span = row('span')
                news = {}
                link = span[2].a.attrs['href']
                news['url'] = link.replace('/news,','').replace('.html','')
                news['titel'] = span[2].a.attrs['title']
                news['author_url'] = span[3].a.attrs['href']
                news['author_name'] = span[3].font.get_text()
                news['ptime'] = span[4].get_text()
                line = json.dumps(dict(news), ensure_ascii=False) + "\n"
                savejson.write(line)
                savecsv.write(news['url'] + ',' + news['ptime'] + ',' + news['titel'] + '\n')
            except:
                pass
    savejson.close
    savecsv.close
    


if __name__ == "__main__":
    datadir = "./data/"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    urlfile = datadir + "url.csv"
    urls = csv.DictReader(open(urlfile, 'r', encoding='utf-8'))
    for item in list(urls):
        datapath = datadir + item['id']
        if not os.path.exists(datapath):
            os.makedirs(datapath)
        savefile = datapath + (datetime.datetime.now().strftime('/%Y-%m-%d'))
        url = 'http://guba.eastmoney.com/list,' + item['id'] + '.html'
        get_urls(url, savefile)
        #clean(savefile)

