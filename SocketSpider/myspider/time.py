#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
import time
 
# 格式化成2016-03-20 11:45:39形式
str = time.strftime("%Y", time.localtime()) + "年" + time.strftime("%m", time.localtime()) + "年" + time.strftime("%d", time.localtime()) + "日 "
print(str)
