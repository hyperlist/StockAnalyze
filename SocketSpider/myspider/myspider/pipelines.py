import codecs
import json
from myspider.items import ArticlesItem,ShortNewsItem

class MyspiderPipeline(object):
    def __init__(self):
        #保存为json
        self.articles = codecs.open('articles.csv', 'a', encoding='utf-8')
        self.shortnews = codecs.open('shortnews.csv', 'a', encoding='utf-8')
        
    def process_item(self, item, spider):
        #print(item)
        if isinstance(item, ArticlesItem):
            #print(item)
            line = json.dumps(dict(item), ensure_ascii=False) + "\n"
            self.articles.write(line)
            try:
                short = {}
                short['id'] = item['id']
                short['time'] = item['time']
                short['review'] = item['review']
                line2 = json.dumps(dict(short), ensure_ascii=False) + "\n"
                self.shortnews.write(line2)
            except Exception as err:
                print(err)
        else:
            #print(item)
            try:
                line = json.dumps(dict(item), ensure_ascii=False) + "\n"
                #print(line)
                self.shortnews.write(line)
            except Exception as err:
                print(err)


    def spider_closed(self, spider):
        self.articles.close()
        self.shortnews.close()