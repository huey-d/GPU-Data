# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import sqlite3

class PcspecsPipeline:

    def __init__(self):
        self.create_connection()
        self.create_table()

    def create_connection(self):
        self.conn = sqlite3.connect('gpu_data.db')
        self.curr = self.conn.cursor()

    def create_table(self):
        self.curr.execute("""DROP TABLE if EXISTS gpu_tb""")
        self.curr.execute('''CREATE TABLE IF NOT EXISTS gpu_tb(
            rank int,
            url varchar,
            productname varchar,
            price int,
            rating varchar,
            brand varchar,
            chipmake varchar
        )''')
        pass

    def process_item(self, item, spider):
        self.store_db(item)
        return item

    def store_db(self, item):
        
        self.curr.execute("""
            INSERT INTO gpu_tb VALUES (?,?,?,?,?,?,?)""", 
            (
            item['rank'],
            item['url'],
            item['productname'],
            item['price'],
            item['rating'],
            item['brand'],
            item['chipmake']))
        self.conn.commit()